import os
import json
import hashlib
import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from utils.rag_utils import (
    load_vectorstore,
    create_conversation_chain,
    build_vectorstore_from_pdf,
)
from utils.rent_tools import (
    calculate_rent,
    calculate_moveout_date,
    get_repair_responsibility,
)

# ============== 初始化配置 ==============
load_dotenv()
st.set_page_config(page_title="Tenant Chat | Smart Rental", page_icon="💬", layout="wide")

# 🚫 完全禁用侧边导航栏
st.markdown("""
    <style>
    /* 隐藏整个侧边导航容器 */
    [data-testid="stSidebarNav"], 
    [data-testid="stSidebarNavLink"],
    [data-testid="stSidebarNavSection"],
    [data-testid="stSidebarHeader"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* 调整侧边栏宽度，只保留自定义的用户信息 */
    [data-testid="stSidebar"] {
        width: 220px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============== 登录校验 ==============
if st.session_state.get("user_role") != "tenants":
    st.warning("Please log in as a tenant to access this page.")
    st.switch_page("app.py")

# ============== 侧边栏：用户信息与配置 ==============
with st.sidebar:
    username = st.session_state.get("username", "Unknown User")
    st.markdown(f"👋 **Current User: {username}**")

    # 登出按钮
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.switch_page("app.py")

    st.markdown("---")

    # API Key 状态
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key")
    if api_key:
        st.success("✅ Detected OpenAI API Key")
    else:
        key_input = st.text_input("🔑 Enter OpenAI API Key", type="password")
        if key_input:
            os.environ["OPENAI_API_KEY"] = key_input
            st.session_state["openai_key"] = key_input
            st.success("✅ API Key saved")

    st.markdown("---")

# ============== 工具函数 ==============
def _file_sha1(uploaded_file) -> str:
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    return hashlib.sha1(data).hexdigest()

def rebuild_pipeline_from_loaded_contracts():
    """修复版的向量库重建函数"""
    vs_values = list(st.session_state.vectorstores_map.values())
    if not vs_values:
        st.session_state.conversation_chain = None
        st.session_state.chain_invoke_safe = None
        st.session_state.agent = None
        st.info("📭 No contracts loaded")
        return

    try:
        # 简单的合并策略：只使用第一个向量库
        if len(vs_values) > 0:
            merged_vs = vs_values[0]
            st.success(f"✅ Loaded contracts with retrieval functionality")
        else:
            st.warning("⚠️ No available vector stores")
            return

        # 重建对话链
        chain, llm, memory, chain_invoke_safe = create_conversation_chain(
            merged_vs, openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        st.session_state.conversation_chain = chain
        st.session_state.chain_invoke_safe = chain_invoke_safe
        st.session_state.llm = llm
        st.session_state.memory = memory
        
        # 重建Agent
        tools = [calculate_rent, calculate_moveout_date, get_repair_responsibility]
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,  # 开启详细日志用于调试
            memory=memory,
            handle_parsing_errors=True
        )
        st.session_state.agent = agent
        st.success("🤖 Agent initialized successfully")
        
    except Exception as e:
        st.error(f"❌ Pipeline reconstruction failed: {e}")

def scroll_to_bottom():
    st_html(
        """
        <script>
            var chatDiv = window.parent.document.querySelector('.main');
            if (chatDiv) { chatDiv.scrollTop = chatDiv.scrollHeight; }
        </script>
        """,
        height=0
    )

# ============== 初始化状态 ==============
defaults = {
    "chat": [],
    "vectorstores_map": {},
    "loaded_keys": set(),
    "conversation_chain": None,
    "chain_invoke_safe": None,
    "agent": None,
    "current_contract_link": None,  # 当前合同的云端链接
    "current_contract_qr": None,    # 当前合同的二维码路径
    "current_contract_id": None     # 当前合同的ID
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 顶部品牌
st.markdown("""
<div style="background:#2E8B57;padding:12px 16px;border-radius:12px;margin-bottom:16px;">
  <h3 style="color:#fff;margin:0;">💬 Tenant Smart Assistant</h3>
</div>
""", unsafe_allow_html=True)

# ============== 左右布局 ==============
col_left, col_right = st.columns([1.15, 2], gap="large")

# ---------------- 左侧：合同管理 ----------------
with col_left:
    st.markdown("### 📂 Contract Management")

    contract_id = st.text_input("Enter Lease ID (e.g., MSH2025-001)")
    if st.button("📥 Load Database Contract", use_container_width=True):
        cid = contract_id.strip()
        if not cid:
            st.info("⚠️ Please enter a lease ID before loading.")
        else:
            db_path = os.path.join("db", cid)
            if not os.path.isdir(db_path):
                st.error("❌ Lease ID not found.")
            else:
                # 检查新合同的云端链接信息
                meta_path = os.path.join(db_path, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        # 如果新合同有云端链接，更新快速访问
                        if meta.get("cloud_link"):
                            st.session_state.current_contract_link = meta["cloud_link"]
                            st.session_state.current_contract_id = cid
                            if meta.get("qr_code"):
                                qr_path = os.path.join(db_path, meta["qr_code"])
                                if os.path.exists(qr_path):
                                    st.session_state.current_contract_qr = qr_path
                            st.success("✅ Quick access link updated")
                        else:
                            # 如果新合同没有云端链接，清除快速访问
                            st.session_state.current_contract_link = None
                            st.session_state.current_contract_id = None
                            st.session_state.current_contract_qr = None
                            if "current_contract_link" in st.session_state and st.session_state.current_contract_link:
                                st.info("ℹ️ New contract has no cloud link, quick access cleared.")
                
                key = f"db:{cid}"
                if key in st.session_state.loaded_keys:
                    st.info("Contract already loaded, skipping.")
                else:
                    with st.spinner("Loading contract..."):
                        try:
                            vs = load_vectorstore(db_path, os.getenv("OPENAI_API_KEY"))
                            st.session_state.vectorstores_map[key] = vs
                            st.session_state.loaded_keys.add(key)
                            rebuild_pipeline_from_loaded_contracts()
                            st.success(f"✅ Database contract loaded: {cid}")
                        except Exception as e:
                            st.error(f"❌ Loading failed: {e}")

    # 显示当前加载合同的云端链接和二维码
    if st.session_state.current_contract_link:
        st.markdown("---")
        st.markdown("### 📱 Current Contract Quick Access")
        st.info(f"**Current Contract: ** {st.session_state.current_contract_id}")
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown(f"**Cloud Link: **\n{st.session_state.current_contract_link}")
        if st.session_state.current_contract_qr:
            with cols[1]:
                st.image(st.session_state.current_contract_qr, caption="Scan to Access Contract")


    st.markdown("---")
    st.subheader("📎 Upload My Contract PDF")
    up = st.file_uploader("Select PDF file", type=["pdf"])
    if up and st.button("📄 Parse PDF Contract", use_container_width=True):
        sha1 = _file_sha1(up)
        up_key = f"upload:{sha1}:{getattr(up, 'size', 0)}"
        if up_key in st.session_state.loaded_keys:
            st.info("This PDF is already loaded, skipping.")
        else:
            with st.spinner("Parsing and building index..."):
                try:
                    vs = build_vectorstore_from_pdf(up, openai_api_key=os.getenv("OPENAI_API_KEY"))
                    st.session_state.vectorstores_map[up_key] = vs
                    st.session_state.loaded_keys.add(up_key)
                    rebuild_pipeline_from_loaded_contracts()
                    st.success(f"✅ Custom contract loaded: {up.name}")
                except Exception as e:
                    st.error(f"❌ Parsing failed: {e}")

   # ---------------- Current Loaded Contracts Display ----------------
    st.markdown("### 📄 Current Loaded Contracts")

    if st.session_state.vectorstores_map:
        delete_keys = []  # 记录要删除的键

        for key in list(st.session_state.vectorstores_map.keys()):
            # 判断来源与显示名
            if key.startswith("db:"):
                source = "📁 Database"
                name = key.replace("db:", "")
            elif key.startswith("upload:"):
                source = "📎 Uploaded File"
                name = f"{key.split(':')[1][:8]}..."  # 用 SHA1 的前几位代替
            else:
                source = "❓ Other"
                name = key

            # 每一行显示
            cols = st.columns([2, 3, 1])
            with cols[0]:
                st.markdown(f"**{source}**")
            with cols[1]:
                st.markdown(name)
            with cols[2]:
                if st.button("❌", key=f"del_{key}", use_container_width=True):
                    delete_keys.append(key)

        # 执行删除操作
        if delete_keys:
            for k in delete_keys:
                if k in st.session_state.vectorstores_map:
                    del st.session_state.vectorstores_map[k]
                st.session_state.loaded_keys.discard(k)
            rebuild_pipeline_from_loaded_contracts()
            st.rerun()

        st.success(f"✅ Current contract count: {len(st.session_state.vectorstores_map)}")
    else:
        st.info("📭 No loaded contracts. You can load from the database or upload a PDF file.")

    # Add debugging information section
    st.markdown("---")
    st.markdown("### 🔧 Debug Information")

    # 显示当前状态
    st.write(f"Loaded contract count: {len(st.session_state.vectorstores_map)}")
    st.write(f"Conversation Chain: {'✅' if st.session_state.conversation_chain else '❌'}")
    st.write(f"Agent: {'✅' if st.session_state.agent else '❌'}")
    st.write(f"Chain Invoke Safe: {'✅' if st.session_state.chain_invoke_safe else '❌'}")
    
    # 测试检索功能按钮
    if st.button("🧪 Test Retrieval", key="test_retrieval"):
        if st.session_state.vectorstores_map:
            vs = list(st.session_state.vectorstores_map.values())[0]
            test_results = vs.similarity_search("Rent", k=2)
            st.success(f"Found {len(test_results)} relevant snippets")
            for i, doc in enumerate(test_results):
                st.write(f"**Result {i+1}:** {doc.page_content[:100]}...")
        else:
            st.warning("No available vector stores to test.")


# ---------------- 右侧：智能问答 ----------------
with col_right:
    # 标题 + 清空按钮
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("### 💬💬 Intelligent Q&A")
    with col2:
        if st.button("🗑🗑️", help="Clear chat history"):
            st.session_state.chat = []
            st.rerun()

    # 聊天内容容器
    chat_container = st.container()
    with chat_container:
        for role, text in st.session_state.chat:
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(text)
    scroll_to_bottom()

    # 固定输入框
    st.markdown(
        """
        <style>
            .stChatInputContainer {
                position: fixed !important;
                bottom: 1rem !important;
                width: 58% !important;
                right: 1rem !important;
                z-index: 999;
                background: white;
                padding-top: 0.5rem;
                border-top: 1px solid #ddd;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    q = st.chat_input("Please enter your question (You can ask directly without loading the contract first.)")

    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        scroll_to_bottom()

        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown("🤔🤔 Thinking...")
            
            # ========== 调试信息 ==========
            debug_info = st.empty()
            debug_info.info(f"""
            **Debug Information**
            - Loaded contracts: {len(st.session_state.vectorstores_map)} items
            - RAG Chain: {'✅' if st.session_state.chain_invoke_safe else '❌'}
            - Agent: {'✅' if st.session_state.agent else '❌'}
            """)

            reply = ""
            contract_info = ""

            # 1️⃣ 尝试 RAG
            if st.session_state.chain_invoke_safe:
                try:
                    debug_info.info("🔍 Processing...")
                    res = st.session_state.chain_invoke_safe({"question": q})
                    contract_info = res.get("answer", "")
                    debug_info.success(f"✅ Retrieval complete, information length: {len(contract_info)}")

                    if res.get("source_documents"):
                        debug_info.write(f"📄 Found {len(res['source_documents'])} relevant document snippets")
                        for i, doc in enumerate(res["source_documents"][:2]):
                            debug_info.write(f"Snippet {i+1}: {doc.page_content[:100]}...")
                except Exception as e:
                    debug_info.error(f"❌ RAG retrieval failed: {e}")
                    contract_info = ""
            else:
                debug_info.warning("⚠️ RAG chain not initialized, using generic response")

            # 2️⃣ 判断是否调用工具
            intent = "NO"
            try:
                debug_info.info("🤖 Analyzing user intent...")
                intent_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Judge whether the user needs to calculate rent, termination time, or maintenance responsibilities. If so, answer YES, otherwise answer NO."),
                    ("human", "{q}")
                ])
                ic_chain = LLMChain(llm=intent_llm, prompt=prompt)
                intent = ic_chain.run({"q": q}).strip().upper()
                debug_info.info(f"Intent analysis result: {intent}")
            except Exception:
                debug_info.warning("Intent analysis failed, defaulting to NO")

            # 3️⃣ 调用 Agent（若需要工具）
            if intent == "YES" and st.session_state.agent:
                try:
                    debug_info.info("🚀 Invoking Agent...")
                    fused = (
                        f"Answer the question based on the following information: \n\n"
                        f"【Contract Information】\n{contract_info}\n\n"
                        f"【User Question】{q}\n\n"
                        f"If calculation is needed, please infer rent, lease term, etc. from the contract and call the appropriate tool."
                    )
                    result = st.session_state.agent.invoke({"input": fused})
                    reply = (
                        result.get("output")
                        or result.get("answer")
                        or result.get("result")
                        or str(result)
                    )
                    reply = reply.split("Observation:")[-1].strip()
                    debug_info.success("✅ Agent response complete")
                except Exception as e:
                    debug_info.error(f"❌ Agent invocation failed: {e}")
                    reply = ""

            # 4️⃣ 无 Agent → 普通 LLM
            if not reply:
                try:
                    debug_info.info("💭 Using generic LLM...")
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a rental assistant. Please provide the final answer directly without showing the thought process."),
                        ("human", f"{q}\n\n(Contract context, if any): {contract_info}")
                    ])
                    ans_chain = LLMChain(llm=llm, prompt=prompt)
                    reply = ans_chain.run({"q": q})
                    debug_info.success("✅ LLM response complete")
                except Exception as e:
                    debug_info.error(f"❌ LLM invocation failed: {e}")
                    reply = "Sorry, I encountered an error while processing your request."

            # 清除调试信息，显示最终回答
            debug_info.empty()
            thinking.empty()
            st.markdown(reply)
            scroll_to_bottom()

        st.session_state.chat.append(("assistant", reply))
        st.rerun()