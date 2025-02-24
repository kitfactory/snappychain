from langchain_core.runnables import RunnableLambda

def dev(devmode:bool=True) -> RunnableLambda:
    """
    開発モードフラグを設定する

    Args:
        devmode (bool): 以降の処理を開発者モードで実行

    Returns:
        RunnableLambda: data 辞書型を受け取り、'prompt' 配列にシステムプロンプトを追加して返却します。
    """
    def inner(data):
        if data != None:
            if devmode:
                data["_dev"] = True
            return data
    return RunnableLambda(inner)



def validate(validate:str)-> RunnableLambda:
    def inner(data):
        pass
    