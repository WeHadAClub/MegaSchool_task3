import os
import json
from typing import TypedDict, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END

# Установите ваш API-ключ Mistral
os.environ["MISTRAL_API_KEY"] = "M6OSvBa27krOBulBhhMqloA5YUnA2Ds7"

llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
llm_json = llm.bind(response_format={"type": "json_object"})

def get_history_str(messages: List[Dict[str, str]]) -> str:
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

observer_prompt = ChatPromptTemplate.from_template("""
Вы - агент-наблюдатель в техническом интервью на позицию {position}, грейд {grade}, с опытом {experience}.
Анализируйте последнее сообщение кандидата: {user_message}
Полная история разговора:
{history}

Задачи:
- Проверьте фактическую точность и выявите галлюцинации или неверные утверждения.
- Оцените качество ответа, уверенность, ясность, честность и вовлеченность.
- Проверьте, не off-topic ли, и предложите перенаправление, если нужно.
- Предложите следующее действие для интервьюера: например, задать конкретный вопрос, ответить на вопрос кандидата, углубить, упростить или завершить, если уместно.
- Адаптируйте сложность вопросов на основе производительности ({difficulty_adjust} из предыдущего).
- Обновите подтвержденные навыки и пробелы в знаниях.

Вывод строго в формате JSON:
{{
  "analysis": "Детальный анализ ответа (1-2 абзаца).",
  "suggestion": "Конкретное предложение для следующего ответа интервьюера (например, 'Задайте вопрос о X' или 'Вежливо исправьте галлюцинацию и спросите Y').",
  "add_confirmed": ["список", "подтвержденных", "навыков/тем"],
  "add_gaps": [{{"topic": "название темы", "correct_answer": "правильное объяснение/факт"}}],
  "soft_note": "Заметка о мягких навыках (например, 'Ясная коммуникация, честен о пределах').",
  "difficulty_adjust": "increase" или "decrease" или "same"
}}
""")

observer_chain = observer_prompt | llm_json

interviewer_prompt = ChatPromptTemplate.from_template("""
Вы - интервьюер, проводящий техническое интервью на {position} {grade} с {experience}.
Будьте профессиональны, вежливы и вовлечены. Адаптируйтесь на основе ввода наблюдателя.

Анализ наблюдателя: {analysis}
Предложение наблюдателя: {suggestion}
Полная история разговора:
{history}

Подумайте шаг за шагом, как сформулировать ваш ответ.
Вывод в этом точном формате:
Thought: Ваша внутренняя мысль о том, как proceed (1-2 предложения).
Response: Сообщение для отправки кандидату.
""")

interviewer_chain = interviewer_prompt | llm

class State(TypedDict):
    messages: List[Dict[str, str]]
    confirmed_skills: List[str]
    knowledge_gaps: List[Dict[str, str]]
    soft_notes: List[str]
    position: str
    grade: str
    experience: str
    turn_id: int
    turns: List[Dict]
    last_observer_out: Dict
    last_internal: str
    last_response: str
    difficulty: str  # Текущий уровень сложности

def observer_node(state: State) -> Dict:
    history_str = get_history_str(state["messages"])
    user_message = state["messages"][-1]["content"] if state["messages"] else "Начальное введение."
    input_data = {
        "position": state["position"],
        "grade": state["grade"],
        "experience": state["experience"],
        "history": history_str,
        "user_message": user_message,
        "difficulty_adjust": state.get("difficulty", "medium")
    }
    output = observer_chain.invoke(input_data)
    out_json = json.loads(output.content)
    return {"last_observer_out": out_json}

def interviewer_node(state: State) -> Dict:
    out_json = state["last_observer_out"]
    analysis = out_json["analysis"]
    suggestion = out_json["suggestion"]
    history_str = get_history_str(state["messages"])
    input_data = {
        "position": state["position"],
        "grade": state["grade"],
        "experience": state["experience"],
        "analysis": analysis,
        "suggestion": suggestion,
        "history": history_str
    }
    output = interviewer_chain.invoke(input_data)
    content = output.content
    thought_start = content.find("Thought:")
    response_start = content.find("Response:")
    thought = content[thought_start + 8:response_start].strip() if thought_start != -1 and response_start != -1 else ""
    response = content[response_start + 9:].strip() if response_start != -1 else content
    internal = f"[Observer]: {analysis} [Interviewer]: {thought}"
    new_messages = state["messages"] + [{"role": "assistant", "content": response}]
    updates = {
        "messages": new_messages,
        "confirmed_skills": state["confirmed_skills"] + out_json.get("add_confirmed", []),
        "knowledge_gaps": state["knowledge_gaps"] + out_json.get("add_gaps", []),
        "soft_notes": state["soft_notes"] + [out_json.get("soft_note", "")],
        "difficulty": out_json.get("difficulty_adjust", state["difficulty"]),
        "last_internal": internal,
        "last_response": response
    }
    return updates

graph = StateGraph(State)
graph.add_node("observer", observer_node)
graph.add_node("interviewer", interviewer_node)
graph.add_edge("observer", "interviewer")
graph.set_entry_point("observer")
graph.set_finish_point("interviewer")
app = graph.compile()

feedback_prompt = ChatPromptTemplate.from_template("""
На основе всей истории интервью: {history}
Подтвержденные навыки: {confirmed}
Пробелы в знаниях: {gaps}
Заметки о мягких навыках: {soft}

Сгенерируйте структурированный финальный фидбек в формате JSON. Это должен быть полезный артефакт для обучения, не просто общие фразы.

{{
  "decision": {{
    "grade": "Уровень кандидата (Junior / Middle / Senior) на основе ответов",
    "hiring_recommendation": "Рекомендация по найму (Hire / No Hire / Strong Hire)",
    "confidence_score": "Уровень уверенности в оценке (0-100%)"
  }},
  "technical_review": {{
    "confirmed_skills": ["Список тем, где кандидат дал точные ответы"],
    "knowledge_gaps": [{{"topic": "Тема, где были ошибки или 'не знаю'", "correct_answer": "Правильный ответ на вопрос, который кандидат завалил"}}]
  }},
  "soft_skills": {{
    "clarity": "Оценка, насколько понятно кандидат излагает мысли (детальное описание)",
    "honesty": "Оценка, пытался ли кандидат выкрутиться/соврать или честно признал незнание (детальное описание)",
    "engagement": "Оценка, задавал ли кандидат встречные вопросы (детальное описание, если это было в сценарии)"
  }},
  "roadmap": ["Список конкретных тем/технологий для подтягивания на основе пробелов. Для каждой темы опционально добавьте ссылки на документацию или статьи (например, 'Тема X: https://example.com')"]
}}

Суммируйте и выведите из предоставленных данных. Для roadmap предложите конкретные ресурсы и ссылки, если они известны или логически вытекают из тем (например, официальная документация Python для базовых тем).
""")

feedback_chain = feedback_prompt | llm_json

# Основное выполнение
team_name = "Team Alpha"  # Измените по необходимости

position = input("Введите позицию (например, Backend Developer): ")
grade = input("Введите грейд (например, Junior): ")
experience = input("Введите опыт (например, Пет-проекты на Django, немного SQL): ")

state: State = {
    "messages": [],
    "confirmed_skills": [],
    "knowledge_gaps": [],
    "soft_notes": [],
    "position": position,
    "grade": grade,
    "experience": experience,
    "turn_id": 1,
    "turns": [],
    "difficulty": "medium"
}

print("Начните интервью, представившись и описав свой опыт.")

while True:
    user_input = input("Кандидат: ")
    if "стоп интервью" in user_input.lower() or "stop interview" in user_input.lower() or "стоп игра" in user_input.lower() or "stop" in user_input.lower():
        break
    state["messages"].append({"role": "user", "content": user_input})
    state = app.invoke(state)
    last_response = state["last_response"]
    print(f"Интервьюер: {last_response}")
    internal = state["last_internal"]
    turn = {
        "turn_id": state["turn_id"],
        "agent_visible_message": last_response,
        "user_message": user_input,
        "internal_thoughts": internal
    }
    state["turns"].append(turn)
    state["turn_id"] += 1

# Генерация финального фидбека
history_str = get_history_str(state["messages"])
confirmed = json.dumps(state["confirmed_skills"])
gaps = json.dumps(state["knowledge_gaps"])
soft = "\n".join(state["soft_notes"])
feedback_output = feedback_chain.invoke({
    "history": history_str,
    "confirmed": confirmed,
    "gaps": gaps,
    "soft": soft
})
final_feedback = json.loads(feedback_output.content)

# Сохранение лога
log = {
    "team_name": team_name,
    "turns": state["turns"],
    "final_feedback": final_feedback
}
with open("interview_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=4)

print("Интервью завершено. Лог сохранен в interview_log.json.")