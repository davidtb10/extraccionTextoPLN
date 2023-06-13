import GPT_J_Module

model = None

def init():
    global model
    model = GPT_J_Module.GPT_J()

def generate_text(question):
    return model.inference(question)
