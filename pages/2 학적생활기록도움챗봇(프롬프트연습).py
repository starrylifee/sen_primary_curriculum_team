import streamlit as st
import json
import os
import pathlib
import toml
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS

# Streamlit의 기본 메뉴와 푸터 숨기기
hide_github_icon = """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK{ display: none; }
    #MainMenu{ visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
"""

st.markdown(hide_github_icon, unsafe_allow_html=True)

# 현재 파일의 디렉토리 경로를 기반으로 secrets.toml 파일 경로 설정
secrets_path = pathlib.Path(__file__).parent.parent / ".streamlit" / "secrets.toml"

# secrets.toml 파일 읽기
try:
    with open(secrets_path, "r") as f:
        secrets = toml.load(f)
except FileNotFoundError:
    st.error(f"Secrets file not found at {secrets_path}. Please ensure the file exists.")
    secrets = {}

# OpenAI API 키 로드
openai_api_key = secrets.get("OPENAI", {}).get("API_KEY")
if not openai_api_key:
    st.error("OpenAI API key is missing. Please ensure it is set in the secrets.toml file.")

# Streamlit 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pdf_chain" not in st.session_state:
        st.session_state["pdf_chain"] = None
    if "pdf_retriever" not in st.session_state:
        st.session_state["pdf_retriever"] = None
    if "custom_prompt" not in st.session_state:
        st.session_state["custom_prompt"] = "파일을 분석해서 질문에 답해주세요."
    if "final_prompt" not in st.session_state:
        st.session_state["final_prompt"] = st.session_state["custom_prompt"]

initialize_session_state()

st.title("2024 학적 · 생활기록 도움 챗봇 🤖")

# 사이드바: 가변형 프롬프트 입력
st.sidebar.header("Custom Prompt")
custom_prompt = st.sidebar.text_area("여기에 프롬프트를 입력하세요", st.session_state["custom_prompt"], height=200)

# 프롬프트 적용 버튼
if st.sidebar.button("프롬프트 적용"):
    st.session_state["final_prompt"] = custom_prompt
    st.sidebar.success("프롬프트가 적용되었습니다.")

# 프롬프트 초기화 버튼
if st.sidebar.button("프롬프트 초기화"):
    st.session_state["custom_prompt"] = ""
    st.session_state["final_prompt"] = ""
    st.sidebar.success("프롬프트가 초기화되었습니다.")

# 현재 사용중인 프롬프트 출력
st.sidebar.markdown("### 현재 사용 중인 프롬프트")
st.sidebar.text(st.session_state["final_prompt"])

# 파일 경로 및 프롬프트 설정
file_paths = [
    "./files/2024 초등 학교생활기록부 기재요령.pdf", 
    "./files/2024 학적업무 도움자료.pdf", 
    "./files/초중등교육법 시행령(대통령령).pdf", 
    "./files/초중등교육법(법률)(제19740호).pdf"
]
selected_model = "gpt-4o-mini"

def add_message(role, message):
    """새로운 대화 메시지 추가."""
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_files(file_paths):
    """사전에 지정된 파일들을 임베딩하여 리트리버 생성."""
    all_docs = []
    for file_path in file_paths:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        # 문서의 각 페이지에 대한 메타데이터 설정
        for i, doc in enumerate(docs):
            doc.metadata = {
                "document_name": pathlib.Path(file_path).name,
                "page_number": i + 1  # 페이지 번호는 1부터 시작
            }
            all_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    return vectorstore.as_retriever()

def create_chain(retriever, prompt_text, model_name):
    """체인 생성."""
    llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=openai_api_key)

    def chain(question):
        # Retrieve the relevant documents and pass them to the LLM along with the question
        context_docs = retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in context_docs])  # Concatenate the content of relevant documents
        
        # 메시지 구성
        messages = [
            SystemMessage(content=prompt_text),
            HumanMessage(content=f"{context}\n\n질문: {question}")
        ]
        output = llm.invoke(messages)
        
        # 텍스트만 반환하도록 처리 (메타데이터 제거)
        if isinstance(output, ChatMessage):
            return output.content, context_docs  # ChatMessage의 content 속성만 반환
        else:
            return str(output), context_docs
    
    return chain

def extract_reference_from_metadata(docs):
    """레퍼런스 정보를 추출하는 함수."""
    # 문서 이름과 페이지 정보 추출
    references = []
    for doc in docs:
        references.append(f"문서: {doc.metadata.get('document_name', '알 수 없음')}, 페이지: {doc.metadata.get('page_number', '알 수 없음')}")
    return " | ".join(references)

def clean_response(response):
    """응답 문자열에서 'content=' 부분을 제거하고 실제 텍스트만 반환"""
    if "content=" in response:
        # 'content=' 이후의 실제 텍스트만 추출하고 메타데이터 부분은 제거
        cleaned_response = response.split("content=", 1)[1].split("response_metadata=", 1)[0]
        # 텍스트에서 불필요한 따옴표와 개행 문자 등을 적절히 처리
        cleaned_response = cleaned_response.replace("\\n", "\n").replace("'", "").replace("\\", "").strip()
        return cleaned_response
    return response.replace("\\n", "\n").replace("'", "").replace("\\", "").strip()

# 파일들을 임베딩하고 검색 기능을 생성
retriever = embed_files(file_paths)
chain = create_chain(retriever, prompt_text=st.session_state["final_prompt"], model_name=selected_model)
st.session_state["pdf_retriever"] = retriever
st.session_state["pdf_chain"] = chain

# 질문 입력과 답변 생성 버튼
user_input = st.text_input("질문을 입력하세요", placeholder="예) 의무교육관리위원회는 어떻게 구성하나? / 정원외 관리의 절차를 알려줘.")
generate_btn = st.button("✨ 답변 생성")

if generate_btn and user_input:
    chain = st.session_state["pdf_chain"]
    if chain:
        with st.spinner("AI 응답을 생성하는 중입니다..."):
            response, context_docs = chain(user_input)
            
            # response를 클린업
            ai_answer = clean_response(response)
            
            # 레퍼런스 추출
            reference = extract_reference_from_metadata(context_docs)
            
            # 출력 레이아웃
            st.markdown("### 답변 및 관련 근거")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**답변**")
                st.markdown(ai_answer)  # 오직 content만 출력되도록 보장
            with col2:
                st.markdown("**관련 근거**")
                st.markdown(reference)

            add_message("user", user_input)
            add_message("assistant", ai_answer)
    else:
        st.error("검색기를 초기화하세요.")
