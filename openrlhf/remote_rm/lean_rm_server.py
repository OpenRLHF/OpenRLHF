import traceback
import logging
import time
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

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    rewards = []
    
    for query in input_text.queries:
        try:
            start_time = time.time()
            logger.info(f"Processing query with statement: {query['formal_statement']}")
            
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
            
    return {"rewards": rewards}

@app.on_event("shutdown")
async def shutdown_event():
    lean4_scheduler.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
