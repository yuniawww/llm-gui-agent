#!/usr/bin/env python3
"""
RAG System Test Script
测试整个RAG系统的功能
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag_retriever import get_knowledge_base_stats, is_knowledge_base_available, get_knowledge_context


def test_rag_system():
    """测试RAG系统"""
    print("🧪 RAG系统功能测试")
    print("=" * 60)
    
    # 1. 检查数据库状态
    print("1️⃣ 检查数据库状态")
    print("-" * 30)
    stats = get_knowledge_base_stats()
    print(f"数据库状态: {stats}")
    
    if not is_knowledge_base_available():
        print("❌ 数据库不可用")
        print("💡 请按以下步骤操作:")
        print("   1. 将PDF文件放入 knowledge_base/ 目录")
        print("   2. 运行: python -m utils.db_builder")
        return False
    
    print("✅ 数据库可用")
    print(f"📊 包含 {stats.get('count', 0)} 个文档块")
    
    # 2. 测试检索功能
    print(f"\n2️⃣ 测试检索功能")
    print("-" * 30)
    
    test_queries = [
        "how to open a folder",
        "快捷键操作",
        "copy paste shortcuts",
        "文件保存方法"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("." * 40)
        
        try:
            context = get_knowledge_context(query)
            if context:
                # 只显示前200个字符
                preview = context[:200] + "..." if len(context) > 200 else context
                print(f"✅ 找到相关信息:")
                print(preview)
            else:
                print("ℹ️  未找到相关信息")
        except Exception as e:
            print(f"❌ 检索失败: {e}")
    
    # 3. 模拟planner集成测试
    print(f"\n3️⃣ 模拟Planner集成测试")
    print("-" * 30)
    
    test_command = "在VS Code中打开一个新文件并编辑"
    print(f"用户命令: {test_command}")
    
    try:
        enhanced_context = get_knowledge_context(test_command)
        
        if enhanced_context:
            print("✅ Planner将获得以下知识库增强信息:")
            print("-" * 20)
            print(enhanced_context[:300] + "..." if len(enhanced_context) > 300 else enhanced_context)
            print("-" * 20)
            print("🎯 这些信息将帮助Planner生成更准确的任务计划")
        else:
            print("ℹ️  该命令没有匹配的知识库信息，Planner将使用基础模式")
            
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
    
    print(f"\n✅ RAG系统测试完成!")
    return True


def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'fitz',  # PyMuPDF
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'fitz':
                import fitz
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True


def main():
    """主函数"""
    print("🚀 RAG系统完整测试")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    print()
    
    # 测试RAG系统
    success = test_rag_system()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 RAG系统测试成功!")
        print("💡 现在可以在Planner中使用知识库增强功能了")
    else:
        print("⚠️  RAG系统需要初始化")
        print("请按照提示完成数据库构建")
    print("=" * 60)


if __name__ == "__main__":
    main()
