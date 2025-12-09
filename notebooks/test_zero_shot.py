from src.baselines.zero_shot_llm import ZeroShotLegalQA


if __name__ == "__main__":
    qa = ZeroShotLegalQA()

    question = "What are the termination conditions in these agreements?"
    result = qa.answer(question)

    print("QUESTION:")
    print(result.question)
    print("\nANSWER:")
    print(result.answer)
