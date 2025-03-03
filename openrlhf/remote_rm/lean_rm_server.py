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
lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=4, timeout=300, memory_limit=10, name='verifier')

# 配置logging为DEBUG级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InputText(BaseModel):
    queries: List[dict]  # 每个query包含formal_statement和proof

class OutputPrediction(BaseModel):
    rewards: List[float]

class DetailedOutputPrediction(BaseModel):
    rewards: List[float]
    details: List[dict]  # 存储每个证明的详细信息

def clean_proof_backticks(proof_text):
    """
    清理证明文本中的多余反引号，特别是末尾的反引号
    """
    # 清理末尾的反引号
    cleaned_text = re.sub(r'\s*```+\s*$', '', proof_text)
    # 清理开头的反引号
    cleaned_text = re.sub(r'^```+\s*', '', cleaned_text)
    # 清理文本中间可能出现的独立反引号
    cleaned_text = re.sub(r'(?<!\n)```(?!\n|[a-zA-Z]+)', '', cleaned_text)
    return cleaned_text

@app.post("/predict")
async def predict(input_text: InputText) -> List[float]:
    rewards = []
    
    for query in input_text.queries:
        try:
            start_time = time.time()
            logger.info(f"Processing query with statement: {query['formal_statement']}")
            
            # 清理proof中可能存在的多余反引号
            if 'proof' in query and query['proof']:
                original_proof = query['proof']
                cleaned_proof = clean_proof_backticks(original_proof)
                if original_proof != cleaned_proof:
                    logger.info("Cleaned extraneous backticks from proof")
                query['proof'] = cleaned_proof
            
            # 创建ProofSummarizer实例
            summarizer = ProofSummarizer(
                data={
                    'formal_statement': query['formal_statement'],
                },
                scheduler=lean4_scheduler
            )
            
            logger.info(f"Analyzing proof: {query['proof']}")
            
            # 分析证明
            proof = summarizer.analyze(
                code=query['proof'],
                require_verification=True
            )
            
            logger.info("Waiting for verification result...")
            # 等待结果
            while not proof.is_result_ready():
                pass
            
            result = proof.result
            logger.info(f"Verification result: {result}")
            
            # 根据验证结果返回reward
            if result.get('complete', False):
                logger.info("Proof is complete and correct")
                rewards.append(1.0)  # 完全正确
            elif result.get('pass', False):
                logger.info("Proof passes but may use sorry")
                rewards.append(0.5)  # 语法正确但可能使用了sorry
            else:
                logger.info(f"Proof has errors: {result.get('errors', [])}")
                rewards.append(-1.0)  # 存在错误
                
            logger.info(f"Verification completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Final rewards: {rewards}")
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)
            
    return rewards

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
                'output': None,  # 添加output字段存储详细错误信息
                'system_messages': None  # 添加system_messages字段
            }
            
            # 清理proof中可能存在的多余反引号
            if 'proof' in query and query['proof']:
                original_proof = query['proof']
                cleaned_proof = clean_proof_backticks(original_proof)
                if original_proof != cleaned_proof:
                    detail['cleaned_proof'] = True
                query['proof'] = cleaned_proof
            
            # 创建ProofSummarizer实例并验证
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
            
            # 更新详细信息
            detail.update({
                'verification_time': verification_time,
                'errors': result.get('errors', []),
                'status': result.get('status', 'unknown'),
                'complete': result.get('complete', False),
                'pass': result.get('pass', False),
                'output': result.get('output', ''),  # 添加详细的错误输出
                'system_messages': result.get('system_messages', '')  # 添加系统消息
            })
            
            # 根据验证结果返回reward
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
