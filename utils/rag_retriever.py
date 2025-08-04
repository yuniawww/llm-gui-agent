#!/usr/bin/env python3
"""
RAG Retriever - 检索相关内容
用途：根据用户查询检索相关文档内容
使用：from utils.rag_retriever import get_knowledge_context
"""

import chromadb
from typing import List, Dict, Optional
import os

# RAG 配置常量
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "./vector_db"


class RAGRetriever:
    """RAG检索器"""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.available = False
        self.query_model = None
        
        try:
            # 检查数据库是否存在
            if not os.path.exists(VECTOR_DB_PATH):
                print("⚠️  向量数据库不存在，请先运行 'python -m utils.db_builder' 构建数据库")
                return
            
            # 设置模型缓存目录到项目文件夹
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(project_root, "embedding_models")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir
            
            # 加载查询用的嵌入模型（与构建时保持一致）
            try:
                from sentence_transformers import SentenceTransformer
                print("📥 加载查询嵌入模型...")
                
                try:
                    self.query_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache_dir)
                    print("✅ Qwen3 查询模型加载成功")
                except Exception as qwen_error:
                    print(f"⚠️  Qwen3 模型加载失败: {qwen_error}")
                    print("🔄 回退到轻量级模型...")
                    self.query_model = SentenceTransformer(FALLBACK_MODEL_NAME, cache_folder=cache_dir)
                    print("✅ 回退查询模型加载成功")
                    
            except Exception as model_error:
                print(f"❌ 查询模型加载失败: {model_error}")
                return
            
            # 连接ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            self.collection = self.chroma_client.get_collection(name="agent_knowledge")
            self.available = True
            print("✅ RAG检索器初始化成功")
            
        except Exception as e:
            print(f"❌ RAG检索器初始化失败: {e}")
            print("💡 请确保已运行 'python -m utils.db_builder' 构建向量数据库")
            self.available = False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if not self.available:
            return []
        
        try:
            # 使用 Qwen3 模型生成查询嵌入
            # 注意：对于查询，使用 prompt_name="query" 以获得更好的效果
            query_embedding = None
            try:
                # 检查模型是否有 prompts 属性（Qwen3 特有）
                if hasattr(self.query_model, 'prompts') and 'query' in self.query_model.prompts:
                    query_embedding = self.query_model.encode([query], prompt_name="query", convert_to_numpy=True)
                else:
                    query_embedding = self.query_model.encode([query], convert_to_numpy=True)
                
                # 使用ChromaDB进行相似度搜索
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=top_k
                )
            except Exception as embed_error:
                print(f"⚠️  嵌入生成失败，使用文本查询: {embed_error}")
                # 回退到文本查询
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
            
            # 格式化结果
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    formatted_results.append({
                        "content": doc,
                        "source": metadata.get("source", "unknown"),
                        "page": metadata.get("page", 0),
                        "relevance_score": max(0, 1 - distance)  # 转换为相似度分数 (0-1)
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 检索过程出错: {e}")
            return []
    
    def format_context(self, search_results: List[Dict], max_length: int = 1000) -> str:
        """
        将搜索结果格式化为上下文字符串
        
        Args:
            search_results: 搜索结果列表
            max_length: 每个结果的最大长度
            
        Returns:
            格式化的上下文字符串
        """
        if not search_results:
            return ""
        
        context_parts = []
        context_parts.append("## 📚 相关知识库信息")
        
        for i, result in enumerate(search_results, 1):
            score = result.get('relevance_score', 0)
            source = result.get('source', 'unknown')
            page = result.get('page', 0)
            content = result.get('content', '')
            
            # 限制内容长度
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            context_parts.append(f"\n### {i}. 来源: {source} (第{page}页)")
            context_parts.append(f"**相关度**: {score:.2f}")
            context_parts.append(f"**内容**: {content}")
        
        context_parts.append("\n---")
        
        return "\n".join(context_parts)
    
    def get_database_info(self) -> Dict:
        """获取数据库信息"""
        if not self.available:
            return {"status": "unavailable", "count": 0}
        
        try:
            count = self.collection.count()
            return {
                "status": "available",
                "count": count,
                "collection_name": "agent_knowledge"
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}


# 全局检索器实例（单例模式）
_retriever = None

def get_retriever() -> RAGRetriever:
    """获取检索器实例（单例模式）"""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever

def search_knowledge(user_command: str, top_k: int = 3) -> str:
    """
    主要接口：根据用户命令检索相关知识
    
    Args:
        user_command: 用户命令/查询
        top_k: 返回结果数量
    
    Returns:
        格式化的上下文字符串，可直接添加到LLM prompt中
    """
    retriever = get_retriever()
    if not retriever.available:
        return ""
    
    search_results = retriever.search(user_command, top_k)
    
    if not search_results:
        return ""
    
    return retriever.format_context(search_results)

def get_knowledge_context(user_command: str) -> str:
    """
    简化的接口，直接返回知识库上下文
    
    Args:
        user_command: 用户命令
        
    Returns:
        知识库上下文字符串
    """
    return search_knowledge(user_command, top_k=3)

def is_knowledge_base_available() -> bool:
    """检查知识库是否可用"""
    retriever = get_retriever()
    return retriever.available

def get_knowledge_base_stats() -> Dict:
    """获取知识库统计信息"""
    retriever = get_retriever()
    return retriever.get_database_info()


if __name__ == "__main__":
    """测试代码"""
    print("🧪 测试RAG检索功能")
    print("=" * 50)
    
    # 检查数据库状态
    stats = get_knowledge_base_stats()
    print(f"📊 数据库状态: {stats}")
    
    if not is_knowledge_base_available():
        print("❌ 知识库不可用，请先运行构建脚本")
        exit(1)
    
    # 测试查询
    test_queries = [
        "how to create a new file in VS Code",
        "如何打开文件夹",
        "VS Code shortcuts",
        "复制粘贴快捷键",
        "如何保存文件"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 40)
        context = get_knowledge_context(query)
        
        if context:
            print(context)
        else:
            print("没有找到相关信息")
    
    print("\n✅ 测试完成")
