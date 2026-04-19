# ================================================================
#  capstone_streamlit.py — HR Policy Bot Streamlit UI
#  Run: streamlit run capstone_streamlit.py
# ================================================================

import os, uuid
from datetime import datetime
from typing import TypedDict, List
import streamlit as st

st.set_page_config(page_title="TechNova HR Bot", page_icon="🏢", layout="centered")

@st.cache_resource
def load_agent():
    """Load all heavy resources once. Cached — never reloads on rerun."""
    from langchain_groq import ChatGroq
    from sentence_transformers import SentenceTransformer
    import chromadb, re
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Run: $env:GROQ_API_KEY='your_key' in terminal first.")
        st.stop()

    MODEL   = "llama-3.3-70b-versatile"
    F_THRESH = 0.7
    MAX_R    = 2
    WINDOW   = 6

    class State(TypedDict):
        question: str; messages: List[dict]; route: str
        retrieved: str; sources: List[str]; tool_result: str
        answer: str; faithfulness: float; eval_retries: int
        employee_name: str

    llm      = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    docs = [
        {"id":"doc_001","topic":"Annual Leave Policy","text":"TechNova Solutions provides all full-time employees with 18 days of paid annual leave per calendar year. Leave is accrued at 1.5 days per month. Employees who join mid-year receive pro-rated leave. Annual leave must be applied at least 3 working days in advance through the HR portal. A maximum of 10 unused leave days can be carried forward. Any leave beyond 10 days lapses at end of December. Employees with more than 5 years of service get 22 days. Probation employees get 9 days only with no carry-forward. Leave without approval is treated as Leave Without Pay (LWP)."},
        {"id":"doc_002","topic":"Sick Leave Policy","text":"TechNova Solutions provides 10 days of paid sick leave per calendar year to all confirmed employees. Sick leave cannot be carried forward or encashed. For 1 to 2 consecutive sick days, inform manager and HR before 10 AM. For sick leave exceeding 2 consecutive days, a medical certificate is mandatory within 3 working days of returning. Sick leave cannot be combined with annual leave. Probation employees get 5 days only. Misuse of sick leave may lead to disciplinary action."},
        {"id":"doc_003","topic":"Maternity and Paternity Leave","text":"Female employees with at least 80 days worked in prior 12 months get 26 weeks paid maternity leave for first two children; 12 weeks for third child onwards. Miscarriage: 6 weeks paid leave. Adoptive mothers: 12 weeks. Male employees get 5 days paid paternity leave within 6 months of birth or adoption. Paternity leave must be taken in one continuous block with supporting documents submitted via HR portal."},
        {"id":"doc_004","topic":"Work From Home Policy","text":"Confirmed employees can work from home up to 2 days per week with manager approval. Physical presence roles (IT infrastructure, lab, reception, security) are not eligible. Employees must be reachable on email, Slack, MS Teams from 9 AM to 6 PM. Attendance must be marked on HR portal by 9:30 AM. Probation employees are not eligible for WFH in the first 3 months. More than 5 consecutive WFH days requires VP approval. Data breach while WFH is a serious disciplinary offence."},
        {"id":"doc_005","topic":"Payroll and Salary Structure","text":"Salary is credited on the last working day of each month via NEFT. Structure: Basic Pay is 40 percent of CTC, HRA is 20 percent of Basic, Special Allowance is 30 percent of Basic, Performance Bonus is 10 percent of CTC paid quarterly. PF deducted at 12 percent of Basic with equal company contribution. TDS deducted per declared tax slab. Investment declarations due by April 30. Salary slips on HR portal by the 5th of the following month. Payroll queries must be raised within 30 days."},
        {"id":"doc_006","topic":"Performance Review and Appraisal","text":"Bi-annual reviews: Mid-Year in July and Annual in March. Ratings 1 to 5: 5 Outstanding, 4 Exceeds Expectations, 3 Meets Expectations, 2 Needs Improvement, 1 Unsatisfactory. Rating 4 or above: 10 to 20 percent increment. Rating 3: 5 to 10 percent. Rating 2: placed on 90-day PIP. Rating 1 twice consecutively may lead to termination. Promotion requires rating 4 or above for two consecutive cycles and 2 years in current role. Increments effective April 1."},
        {"id":"doc_007","topic":"Resignation and Exit Policy","text":"Submit formal resignation to manager and HR. Notice period: under 1 year = 30 days; 1 to 3 years = 60 days; over 3 years = 90 days. Notice buyout allowed with approval: Basic Pay divided by 30 multiplied by days being bought out. Full and Final settlement within 45 working days including salary, leave encashment up to 30 days, and expense reimbursements. Company assets returned on last working day. Relieving letter and experience certificate issued after full clearance."},
        {"id":"doc_008","topic":"Travel and Expense Reimbursement","text":"Air travel needs department head approval booked via company travel portal. Business class only for Director level and above. Hotel reimbursed up to Rs 5000 per night in Tier 1 cities (Mumbai, Delhi, Bangalore, Chennai, Hyderabad) and Rs 3000 in Tier 2 cities. Daily food allowance Rs 500. Local conveyance on actuals with receipts. Personal vehicle at Rs 12 per km. Claims within 15 days of return. Claims after 30 days not processed. Alcohol and personal expenses not reimbursable."},
        {"id":"doc_009","topic":"Code of Conduct and Disciplinary Policy","text":"Employees must treat colleagues with respect, maintain confidentiality, not engage in harassment or discrimination, not misuse company resources. Disciplinary process: Stage 1 verbal warning, Stage 2 written warning, Stage 3 termination with cause. Serious offences (fraud, theft, sexual harassment, data breach, violence) lead to immediate termination without notice. GRC available for unresolved complaints."},
        {"id":"doc_010","topic":"Employee Benefits and Perquisites","text":"Health Insurance: Group mediclaim covering employee, spouse, two children, and dependent parents up to Rs 5 lakh per annum. Top-up up to Rs 10 lakh at subsidised premium. Term Life Insurance: 3 times annual CTC at no cost. Gratuity after 5 years: 15 times Last Basic Salary times Years of Service divided by 26. ESOP for Senior Engineer and above with 4-year vesting. Meal card Rs 2200 per month. Gym reimbursement up to Rs 1500 per month. Learning and Development Rs 20000 per year."},
    ]

    client = chromadb.Client()
    try: client.delete_collection("hr_kb_ui")
    except: pass
    col = client.create_collection("hr_kb_ui")
    texts = [d["text"] for d in docs]
    col.add(documents=texts, embeddings=embedder.encode(texts).tolist(),
            ids=[d["id"] for d in docs], metadatas=[{"topic": d["topic"]} for d in docs])

    def mem(s):
        msgs = s.get("messages",[]) + [{"role":"user","content":s["question"]}]
        msgs = msgs[-WINDOW:]
        name = s.get("employee_name","")
        if "my name is" in s["question"].lower():
            after = s["question"].lower().split("my name is")[-1].strip()
            if after: name = after.split()[0].capitalize()
        return {"messages":msgs,"employee_name":name,"eval_retries":0,"tool_result":""}

    def router(s):
        hist = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in s.get("messages",[])[-4:])
        p = (f"Router for TechNova HR chatbot. Reply ONE WORD: retrieve, tool, or memory_only.\n"
             f"- retrieve: any HR policy, leave, salary, WFH, appraisal, benefits, resignation, travel, conduct\n"
             f"- tool: question needs today's date\n"
             f"- memory_only: greeting or answer already in history\n"
             f"History:\n{hist}\nQuestion: {s['question']}\nRoute:")
        r = llm.invoke(p).content.strip().lower().split()[0]
        if r not in ("retrieve","tool","memory_only"): r = "retrieve"
        return {"route": r}

    def retrieve(s):
        qe = embedder.encode([s["question"]]).tolist()
        res = col.query(query_embeddings=qe, n_results=3)
        parts, sources = [], []
        for chunk, meta in zip(res["documents"][0], res["metadatas"][0]):
            parts.append(f"[{meta['topic']}]\n{chunk}"); sources.append(meta["topic"])
        return {"retrieved": "\n\n".join(parts), "sources": sources}

    def skip(s):
        return {"retrieved":"","sources":[]}

    def tool(s):
        try:
            now = datetime.now()
            tr = f"Today is {now.strftime('%A, %d %B %Y')}. Time: {now.strftime('%I:%M %p')}."
        except Exception as e: tr = f"Error: {e}"
        return {"tool_result":tr,"retrieved":"","sources":[]}

    def answer(s):
        hist = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in s.get("messages",[])[:-1])
        ni = f"Address the employee as {s.get('employee_name','')}." if s.get("employee_name") else ""
        ri = "Use ONLY information from the context below." if s.get("eval_retries",0)>=1 else ""
        sys_p = (f"You are a professional HR Policy assistant for TechNova Solutions Pvt. Ltd.\n"
                 f"STRICT RULE: Answer ONLY from the CONTEXT below. No general knowledge.\n"
                 f"If not in context: 'I do not have that policy information. Please contact hr@technova.com'\n"
                 f"{ni} {ri}")
        ctx = ""
        if s.get("retrieved"): ctx += f"\n\nHR POLICY CONTEXT:\n{s['retrieved']}"
        if s.get("tool_result"): ctx += f"\n\nCURRENT DATE:\n{s['tool_result']}"
        msg = f"History:\n{hist}{ctx}\n\nEmployee: {s['question']}\nAnswer:"
        return {"answer": llm.invoke(f"{sys_p}\n\n{msg}").content.strip()}

    def ev(s):
        ret = s.get("retrieved",""); retries = s.get("eval_retries",0)
        if not ret: return {"faithfulness":1.0,"eval_retries":retries}
        p = (f"Score faithfulness 0.0-1.0. Does ANSWER use ONLY CONTEXT?\n"
             f"CONTEXT:\n{ret}\nANSWER:\n{s.get('answer','')}\n"
             f"Reply with a decimal number only:")
        try:
            raw = llm.invoke(p).content.strip()
            m = __import__("re").search(r"[0-9]+\.?[0-9]*", raw)
            faith = max(0.0, min(1.0, float(m.group()))) if m else 0.5
        except: faith = 0.5
        return {"faithfulness":faith,"eval_retries":retries+1}

    def save(s):
        return {"messages": s.get("messages",[]) + [{"role":"assistant","content":s.get("answer","")}]}

    def route_dec(s):
        r = s.get("route","retrieve")
        return "tool" if r=="tool" else ("skip" if r=="memory_only" else "retrieve")

    def eval_dec(s):
        return "save" if (s.get("faithfulness",1.0)>=F_THRESH or s.get("eval_retries",0)>=MAX_R) else "answer"

    g = StateGraph(State)
    for name, fn in [("memory",mem),("router",router),("retrieve",retrieve),
                     ("skip",skip),("tool",tool),("answer",answer),("eval",ev),("save",save)]:
        g.add_node(name, fn)
    g.set_entry_point("memory")
    g.add_edge("memory","router"); g.add_edge("retrieve","answer")
    g.add_edge("skip","answer"); g.add_edge("tool","answer")
    g.add_edge("answer","eval"); g.add_edge("save",END)
    g.add_conditional_edges("router",route_dec,{"retrieve":"retrieve","skip":"skip","tool":"tool"})
    g.add_conditional_edges("eval",eval_dec,{"answer":"answer","save":"save"})
    return g.compile(checkpointer=MemorySaver())

# ── Load ──────────────────────────────────────────────────────
app = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages  = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 TechNova HR Bot")
    st.markdown("---")
    st.markdown("**Ask me about:**")
    topics = ["📅 Annual & Sick Leave","🤱 Maternity / Paternity Leave",
              "🏠 Work From Home Policy","💰 Payroll & Salary",
              "⭐ Performance Appraisal","📝 Resignation & Notice Period",
              "✈️ Travel Reimbursement","🤝 Code of Conduct","🎁 Employee Benefits","📆 Date & Deadlines"]
    for t in topics:
        st.markdown(f"- {t}")
    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")

# ── Main ──────────────────────────────────────────────────────
st.title("🏢 TechNova HR Policy Bot")
st.caption("Your 24/7 HR assistant — ask anything about company policies")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("👋 Hello! I'm the TechNova HR Policy Bot. Ask me anything about **leave, salary, WFH, appraisals, resignation, travel, or benefits**. How can I help you today?")

if prompt := st.chat_input("Ask your HR policy question here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Checking HR policies..."):
            result = app.invoke({"question": prompt},
                                config={"configurable":{"thread_id":st.session_state.thread_id}})
            answer = result.get("answer","Something went wrong. Please try again.")
        st.markdown(answer)
        sources = result.get("sources",[])
        if sources:
            with st.expander("📋 Policy sections used"):
                for s in sources: st.markdown(f"- {s}")

    st.session_state.messages.append({"role":"assistant","content":answer})