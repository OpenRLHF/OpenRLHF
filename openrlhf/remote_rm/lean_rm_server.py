import traceback
import logging
import time
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from openrlhf.remote_rm.ds_prover.lean.verifier import Lean4ServerScheduler
from openrlhf.remote_rm.ds_prover.lean.proof import ProofSummarizer
from openrlhf.remote_rm.ds_prover.utils import AttrDict

# 初始化FastAPI应用
app = FastAPI()

# 配置验证器
lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=300, memory_limit=10, name='verifier')

# 配置logging为DEBUG级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InputText(BaseModel):
    queries: List[dict]  # Contains formal_statement and proof

class OutputPrediction(BaseModel):
    rewards: List[float]

class DetailedOutputPrediction(BaseModel):
    rewards: List[float]
    details: List[dict]  # Detailed verification info

def clean_proof_backticks(proof_text):
    """Clean extra backticks from proof text"""
    cleaned_text = re.sub(r'\s*```+\s*$', '', proof_text)
    cleaned_text = re.sub(r'^```+\s*', '', cleaned_text)
    cleaned_text = re.sub(r'(?<!\n)```(?!\n|[a-zA-Z]+)', '', cleaned_text)
    return cleaned_text

def parse_error_positions(error_output, proof_content):
    """Parse error positions and convert to proof-relative positions"""
    error_positions = []
    if not error_output:
        return error_positions
    
    HEADER_LINES = 11  # Fixed header lines in test file
        
    for line in error_output.split('\n'):
        match = re.match(r'.*?:(\d+):(\d+):\s*error:\s*(.*)', line)
        if match:
            file_line = int(match.group(1))
            error_column = int(match.group(2))
            error_message = match.group(3)
            
            proof_line = file_line - HEADER_LINES
            
            if proof_line > 0:  
                proof_lines = proof_content.split('\n')
                if 0 <= proof_line - 1 < len(proof_lines):
                    relative_pos = {
                        'line': proof_line,
                        'file_line': file_line,
                        'column': error_column,
                        'position': sum(len(l) + 1 for l in proof_lines[:proof_line-1]) + error_column,
                        'message': error_message,
                        'content': proof_lines[proof_line - 1]
                    }
                    error_positions.append(relative_pos)
                    logger.debug(f"Error at proof line {proof_line} (file line {file_line}): {error_message}")
                
    return error_positions

@app.post("/predict")
async def predict(input_text: InputText) -> OutputPrediction:
    rewards = []

    for query in input_text.queries:
        try:
            start_time = time.time()
            logger.info(f"Processing query with statement: {query['formal_statement']}")

            if 'proof' in query and query['proof']:
                original_proof = query['proof']
                cleaned_proof = clean_proof_backticks(original_proof)
                if original_proof != cleaned_proof:
                    logger.info("Cleaned extraneous backticks from proof")
                query['proof'] = cleaned_proof

            summarizer = ProofSummarizer(
                data={
                    'formal_statement': query['formal_statement'],
                },
                scheduler=lean4_scheduler
            )

            logger.info(f"Analyzing proof: {query['proof']}")

            proof = summarizer.analyze(
                code=query['proof'],
                require_verification=True
            )

            logger.info("Waiting for verification result...")
            while not proof.is_result_ready():
                pass

            result = proof.result
            logger.info(f"Verification result: {result}")

            if result.get('complete', False):
                logger.info("Proof is complete and correct")
                rewards.append(1.0)  
            elif result.get('pass', False):
                logger.info("Proof passes but may use sorry")
                rewards.append(0.5)  
            else:
                logger.info(f"Proof has errors: {result.get('errors', [])}")
                rewards.append(-1.0)  

            logger.info(f"Verification completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Final rewards: {rewards}")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)

    return {"rewards": rewards}

@app.post("/predict_detail", response_model=DetailedOutputPrediction)
async def predict_detail(input_text: InputText):
    rewards = []
    details = []
    
    for query in input_text.queries:
        try:
            start_time = time.time()
            detail = {
                'formal_statement': query['formal_statement'],
                'verification_time': None,
                'errors': None,
                'status': None,
                'complete': False,
                'pass': False,
                'output': None,  
                'system_messages': None  
            }
            
    
            if 'proof' in query and query['proof']:
                original_proof = query['proof']
                cleaned_proof = clean_proof_backticks(original_proof)
                if original_proof != cleaned_proof:
                    detail['cleaned_proof'] = True
                query['proof'] = cleaned_proof
            
            summarizer = ProofSummarizer(
                data={'formal_statement': query['formal_statement']},
                scheduler=lean4_scheduler
            )
            
            proof = summarizer.analyze(
                code=query['proof'],
                require_verification=True
            )
            
            while not proof.is_result_ready():
                pass
            
            result = proof.result
            verification_time = time.time() - start_time
            
            error_positions = parse_error_positions(
                result.get('output', ''), 
                query['proof']  
            )
            
            detail.update({
                'verification_time': verification_time,
                'errors': result.get('errors', []),
                'status': result.get('status', 'unknown'),
                'complete': result.get('complete', False),
                'pass': result.get('pass', False),
                'output': result.get('output', ''),
                'error_positions': error_positions,  
                'proof_segments': proof.segmentation(result)  
            })
            
            if result.get('complete', False):
                rewards.append(1.0)
            elif result.get('pass', False):
                rewards.append(0.5)
            else:
                rewards.append(-1.0)
                
            details.append(detail)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)
            details.append({
                'formal_statement': query['formal_statement'],
                'verification_time': None,
                'errors': [str(e)],
                'status': 'error',
                'complete': False,
                'pass': False
            })
            
    return {"rewards": rewards, "details": details}

@app.on_event("shutdown")
async def shutdown_event():
    lean4_scheduler.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
