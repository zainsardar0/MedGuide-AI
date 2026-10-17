from app.agent import get_graph, close_graph


graph = get_graph()
config = {"configurable": {"thread_id": "user_123"}}

result = graph.invoke(
    {
        "question": "What causes headaches?",
        "history": [],
        "image_description": "",
        "kb": [],
        "web": [],
        "answer": "",
        "route": "",
        "intent": "",
        "has_image": False,
        "validation_flags": []
    },
    config=config
)

print("Final Result:", result)
close_graph()
