import sys
import codeingest.chain as chain
import codeingest.environ as environ

def get_question():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return input("Enter a question: ")

def main():
    environ.openai()
    environ.langsmith() 

    c = chain.make_chain()
    q = get_question()
    print(c.invoke(q))


if __name__ == "__main__":
    main()
