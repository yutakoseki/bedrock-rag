import streamlit as st
import uuid
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.output_parsers import StrOutputParser

# 検索手段を指定
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="MBYZJIEVHU",  # ここにナレッジベースIDを記載する
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}}
)

# セッションIDを動的に生成
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_id_{uuid.uuid4()}"
    print("st.session_state.session_id : " + st.session_state.session_id)

# セッションに会話履歴を定義
if "history" not in st.session_state:
    st.session_state.history = DynamoDBChatMessageHistory(
        table_name="bsc_db", session_id=st.session_state.session_id
    )

# retrieverの結果を整形する関数
def format_docs(docs):
    return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

# セッションにLangChainの処理チェーンを定義
if "chain" not in st.session_state:
    # プロンプトを定義
    prompt = ChatPromptTemplate.from_messages([
        ("system", """絵文字入りでフレンドリーに会話してください。
        以下の情報のみを使用して回答してください。この情報に含まれていないことについては、「その情報は私のデータベースにありません」と答えてください：

        {context}

        回答する際は、与えられた情報のみを使用し、推測や創造的な解釈は避けてください。"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    # チャット用LLMを定義
    chat = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={
            "max_tokens": 4000,
            "temperature": 0.2,  # より低い温度で、より保守的な回答を促す
        },
        streaming=True,
    )

    # チェーンを定義
    st.session_state.chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "history": lambda x: x["history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | chat
        | StrOutputParser()
    )

# タイトルを画面表示
st.title("Bedrock FAQ")

# 履歴クリアボタンを画面表示
if st.button("履歴クリア"):
    st.session_state.history.clear()

# メッセージを画面表示
for message in st.session_state.history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# チャット入力欄を定義
if prompt := st.chat_input("質問を入力"):
    # ユーザーの入力をメッセージに追加
    with st.chat_message("user"):
        st.markdown(prompt)

    # モデルの呼び出しと結果の画面表示
    with st.chat_message("assistant"):
        response = st.write_stream(
            st.session_state.chain.stream(
                {
                    "history": st.session_state.history.messages,
                    "question": prompt,
                },
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
        )

    # 会話を履歴に追加
    st.session_state.history.add_user_message(prompt)
    st.session_state.history.add_ai_message(response)