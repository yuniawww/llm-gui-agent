#!/usr/bin/env python3
"""
DB Builder - 构建向量数据库
用途：一次性处理knowledge_base/下的PDF文件，构建向量数据库
使用：python -m utils.db_builder  # 运行一次即可
"""

import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import sys

# RAG 配置常量
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "./vector_db"
CHUNK_MIN_LENGTH = 50
BATCH_SIZE = 100


class DBBuilder:
    def __init__(self, cache_dir: str = None):
        print("🚀 初始化数据库构建器...")
        
        # 设置模型缓存目录到项目文件夹
        if not cache_dir:
            # 默认使用项目目录下的 embedding_models 文件夹
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(project_root, "embedding_models")
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
        print(f"📁 使用缓存目录: {cache_dir}")
        
        # 检查GPU可用性
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  使用设备: {device}")
        if device == "cuda":
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 使用 Qwen3 嵌入模型
        try:
            print("📥 加载 Qwen3 嵌入模型...")
            
            # 尝试加载 Qwen3-Embedding-0.6B
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, 
                                                         cache_folder=cache_dir,
                                                         device=device)
                print(f"✅ Qwen3-Embedding-0.6B 加载成功 (设备: {device})")
            except Exception as qwen_error:
                print(f"⚠️  Qwen3 模型加载失败: {qwen_error}")
                print("🔄 回退到轻量级模型...")
                self.embedding_model = SentenceTransformer(FALLBACK_MODEL_NAME, 
                                                         cache_folder=cache_dir,
                                                         device=device)
                print(f"✅ 回退模型加载成功 (设备: {device})")
                
        except Exception as e:
            print(f"❌ 所有嵌入模型加载失败: {e}")
            sys.exit(1)
        
        # 初始化ChromaDB
        try:
            print("🗄️ 初始化ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            
            # 如果集合已存在，删除并重新创建
            try:
                self.chroma_client.delete_collection(name="agent_knowledge")
                print("🔄 删除已存在的数据库集合")
            except:
                pass  # 集合不存在，忽略错误
            
            # 使用简单集合创建，手动控制embedding生成（避免ChromaDB内部调用死机）
            self.collection = self.chroma_client.create_collection(
                name="agent_knowledge"
                # 不指定embedding_function，我们手动生成并传入embedding
            )
            print("✅ ChromaDB初始化成功")
        except Exception as e:
            print(f"❌ ChromaDB初始化失败: {e}")
            sys.exit(1)
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """从PDF提取文本并分块"""
        print(f"📄 处理PDF: {os.path.basename(pdf_path)}")
        
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # 简单分块：按段落分割
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for i, paragraph in enumerate(paragraphs):
                    # 过滤太短的文本，保留有意义的内容
                    if len(paragraph) > CHUNK_MIN_LENGTH:  
                        chunks.append({
                            "text": paragraph,
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "chunk_id": f"{os.path.basename(pdf_path)}_p{page_num+1}_c{i+1}"
                        })
            
            doc.close()
            print(f"  ✅ 提取了 {len(chunks)} 个文本块")
            return chunks
            
        except Exception as e:
            print(f"  ❌ 处理PDF失败: {e}")
            return []
    
    def build_database(self):
        """构建完整的向量数据库"""
        print("🏗️ 开始构建向量数据库...")
        
        knowledge_base_dir = "./knowledge_base"
        
        # 查找PDF文件
        pdf_files = [f for f in os.listdir(knowledge_base_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("❌ 在 knowledge_base/ 目录中没有找到PDF文件")
            print("请将PDF文件放入该目录后重新运行")
            return
        
        print(f"📚 发现 {len(pdf_files)} 个PDF文件:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        
        all_chunks = []
        
        # 处理所有PDF文件
        for filename in pdf_files:
            pdf_path = os.path.join(knowledge_base_dir, filename)
            chunks = self.extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print("❌ 没有提取到任何文本内容")
            return
        
        print(f"\n📊 总共需要处理的文本块: {len(all_chunks)}")
        
        # 准备数据
        texts = [chunk["text"] for chunk in all_chunks]
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        metadatas = [{"source": chunk["source"], "page": chunk["page"]} 
                    for chunk in all_chunks]
        
        print("🔄 生成嵌入向量并存储到数据库...")
        
        try:
            # GPU模式可以使用更大的批量大小
            import torch
            if torch.cuda.is_available():
                batch_size = 16  # GPU模式：大批量
                print("🚀 GPU模式：使用大批量处理")
            else:
                batch_size = 8  # CPU模式：小批量
                print("💻 CPU模式：使用小批量处理")
            
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f"📊 将分 {total_batches} 批处理，每批 {batch_size} 个文本块")
            
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                batch_texts = texts[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                print(f"  🔄 处理第 {batch_num}/{total_batches} 批...")
                
                # GPU模式下可以启用进度条和优化参数
                if torch.cuda.is_available():
                    # GPU模式：启用混合精度和优化
                    embeddings = self.embedding_model.encode(
                        batch_texts, 
                        batch_size=32,  # 内部批量大小
                        show_progress_bar=False,  # 避免过多输出
                        convert_to_numpy=True,
                        normalize_embeddings=True  # 标准化向量
                    )
                else:
                    # CPU模式：保守参数
                    embeddings = self.embedding_model.encode(
                        batch_texts,
                        batch_size=8,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                
                # 存储到ChromaDB
                self.collection.add(
                    documents=batch_texts,
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                    embeddings=embeddings.tolist()
                )
                
                print(f"  ✅ 批次 {batch_num} 完成 ({min(i+batch_size, len(texts))}/{len(texts)})")
                
                # GPU模式下显示内存使用情况
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"     💾 GPU内存: {memory_used:.2f}GB / {memory_total:.1f}GB")
            
            print(f"\n🎉 数据库构建完成!")
            print(f"✅ 成功添加 {len(all_chunks)} 个文本块")
            print(f"💾 数据库保存位置: ./vector_db")
            print(f"📋 集合名称: agent_knowledge")
            
            # 显示统计信息
            print(f"\n📈 统计信息:")
            for pdf_file in pdf_files:
                file_chunks = [c for c in all_chunks if c["source"] == pdf_file]
                print(f"  - {pdf_file}: {len(file_chunks)} 个文本块")
            
        except Exception as e:
            print(f"❌ 数据库构建失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("=" * 60)
    print("🔧 Agent Knowledge Base Builder")
    print("=" * 60)
    
    # 检查是否指定了自定义缓存目录
    import argparse
    parser = argparse.ArgumentParser(description="构建Agent知识库")
    parser.add_argument("--cache-dir", type=str, 
                       help="指定模型缓存目录 (默认: ./embedding_models)")
    args = parser.parse_args()
    
    builder = DBBuilder(cache_dir=args.cache_dir)
    builder.build_database()
    
    print("\n" + "=" * 60)
    print("✨ 构建完成! 现在可以使用RAG检索功能了")
    print("💡 下一步: 在planner中使用知识库增强任务规划")
    print("=" * 60)


if __name__ == "__main__":
    main()
