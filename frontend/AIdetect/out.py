# output from backend
def out(text):
    # Your logic to determine if text is real or fake
    if len(text) > 10:
        print("true")
        return "real"
    else:
        print("false")
        return "fake"