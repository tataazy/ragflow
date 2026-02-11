from deepdoc.parser.pdf_parser import RAGFlowPdfParser
import os

# 替换为实际的PDF文件路径
test_pdf_path = "path/to/your/test.pdf"

if os.path.exists(test_pdf_path):
    parser = RAGFlowPdfParser()
    print(f"Testing PDF parser on: {test_pdf_path}")
    
    # 测试默认配置
    text, tbls = parser(test_pdf_path)
    chunks = text.split("\n\n")
    print(f"Default configuration - Number of chunks: {len(chunks)}")
    print(f"Default configuration - Number of tables: {len(tbls)}")
    
    # 测试禁用表格自动旋转
    text_no_rotate, tbls_no_rotate = parser(test_pdf_path, auto_rotate_tables=False)
    chunks_no_rotate = text_no_rotate.split("\n\n")
    print(f"No auto-rotate configuration - Number of chunks: {len(chunks_no_rotate)}")
    print(f"No auto-rotate configuration - Number of tables: {len(tbls_no_rotate)}")
else:
    print(f"Error: Test PDF file not found at {test_pdf_path}")
    print("Please replace 'path/to/your/test.pdf' with an actual PDF file path.")
