import streamlit as st
from langchain_openai import OpenAIEmbeddings

# OpenAI API 키 입력
openai_api_key = st.text_input("OpenAI API Key를 입력하세요:", type="password")

# 버튼 클릭 시 테스트 실행
if st.button("테스트 실행"):
    if not openai_api_key:
        st.error("API Key를 입력하세요.")
    else:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            st.write("Embedding 객체 초기화 성공!")
        except Exception as e:
            st.error(f"Embedding 객체 초기화 중 오류 발생: {e}")
