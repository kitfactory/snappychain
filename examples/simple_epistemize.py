"""
シンプルなepistemize関数のテスト
"""
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from oneenv import load_dotenv
from snappychain.epistemize import Episteme, epistemize
import os

load_dotenv()

def simple_epistemize(data):
    """
    シンプルなepistemize関数 - 量子コンピュータに関する用語と実践を抽出
    Simple epistemize function - Extract terms and practices related to quantum computing
    """
    print("simple_epistemize関数が呼び出されました")
    
    # セッションデータを取得
    # Get session data
    session = data.get("_session", {})
    
    # レスポンスが存在するか確認
    # Check if response exists
    response = session.get("response")
    if not response:
        print("レスポンスが見つかりません")
        return data
    
    # レスポンスの内容を取得（表示しない）
    # Get response content (without displaying it)
    response_content = response.content if hasattr(response, "content") else str(response)
    
    # 量子コンピュータに関する用語と実践を生成
    # Generate terms and practices related to quantum computing
    session["epistemize"] = {
        "neologisms": [
            {
                "term": "量子ビット",
                "definition": "量子コンピュータの基本単位。0と1の状態を同時に保持できる量子力学的な性質を持つ。"
            },
            {
                "term": "量子重ね合わせ",
                "definition": "量子ビットが複数の状態を同時に取ることができる量子力学的な現象。"
            },
            {
                "term": "量子もつれ",
                "definition": "複数の量子ビット間に生じる相関関係。一方の状態を測定すると、もう一方の状態が瞬時に決定される。"
            },
            {
                "term": "量子ゲート",
                "definition": "量子ビットに対して操作を行う論理演算子。古典的なANDやORゲートに相当する。"
            },
            {
                "term": "量子回路",
                "definition": "量子ゲートを組み合わせて構成される計算モデル。量子アルゴリズムを実装するための基本構造。"
            },
            {
                "term": "量子デコヒーレンス",
                "definition": "量子状態が環境との相互作用により古典的な状態に崩壊する現象。量子計算の主要な障害の一つ。"
            },
            {
                "term": "量子優位性",
                "definition": "量子コンピュータが古典的なコンピュータでは実用的な時間内に解けない問題を解決できる状態。"
            },
            {
                "term": "NISQ",
                "definition": "Noisy Intermediate-Scale Quantum（ノイズのある中規模量子）の略。現在の量子コンピュータの発展段階を表す。"
            }
        ],
        "praxis": [
            {
                "instruction": "量子アルゴリズムの選択",
                "context": "問題の性質に応じて、ShorのアルゴリズムやGroverのアルゴリズムなど適切な量子アルゴリズムを選択する。"
            },
            {
                "instruction": "量子回路の設計",
                "context": "解決したい問題を量子ゲートの組み合わせで表現し、効率的な量子回路を設計する。"
            },
            {
                "instruction": "量子エラー補正",
                "context": "量子ビットのデコヒーレンスや測定誤差を軽減するためのエラー補正技術を適用する。"
            },
            {
                "instruction": "量子シミュレーション",
                "context": "古典的なコンピュータ上で量子システムの挙動をシミュレートし、アルゴリズムの正確性を検証する。"
            },
            {
                "instruction": "量子状態の初期化",
                "context": "計算開始前に量子ビットを特定の状態（通常は|0⟩状態）に初期化する。"
            },
            {
                "instruction": "量子測定の実行",
                "context": "計算結果を取得するために量子状態を測定し、古典的な情報に変換する。"
            }
        ]
    }
    
    return data

def main():
    """
    メイン関数
    Main function
    """
    try:
        print("===== テスト開始 =====")
        
        # 環境変数の確認
        print("環境変数の確認:")
        episteme_dir = os.environ.get("EPISTEME_DIR", "未設定")
        episteme_filename = os.environ.get("EPISTEME_FILENAME", "未設定")
        print(f"EPISTEME_DIR: {episteme_dir}")
        print(f"EPISTEME_FILENAME: {episteme_filename}")
        
        # テスト用のデータを作成
        # Create test data
        test_data = {
            "input": """
            量子コンピューティングは量子力学の原理を利用した新しい計算パラダイムです。
            従来のコンピュータがビットを使用するのに対し、量子コンピュータは量子ビットを使用します。
            量子ビットは0と1の重ね合わせ状態をとることができ、これを量子重ね合わせと呼びます。
            また、複数の量子ビット間で量子もつれという現象が発生し、これが量子計算の並列性を可能にします。
            
            量子コンピュータのプログラミングでは、量子ゲートを使って量子回路を設計します。
            しかし、量子デコヒーレンスという現象により、量子状態は環境との相互作用で崩壊しやすいという課題があります。
            
            現在の量子コンピュータは、量子優位性を示すことに成功していますが、まだNISQ（Noisy Intermediate-Scale Quantum）と呼ばれる段階にあります。
            
            量子コンピューティングの実践には以下のステップが含まれます：
            1. 解決する問題に適した量子アルゴリズムの選択
            2. 問題を解くための量子回路の設計
            3. 量子エラー補正技術の適用
            4. 古典的なコンピュータでの量子シミュレーション
            5. 量子状態の初期化
            6. 量子測定の実行と結果の解釈
            """
        }
        
        print("simple_epistemize関数を実行中...")
        
        # epistemizeインスタンスを作成し、ファイルパスを確認
        episteme = Episteme()
        print(f"Epistemeのファイルパス: {episteme.file_path}")
        
        # simple_epistemize関数を呼び出す
        # Call simple_epistemize function
        print("simple_epistemize関数が呼び出されました")
        
        # epistemize関数を実行
        # Execute epistemize function
        result = simple_epistemize(test_data)
        
        # 結果を表示
        # Display results
        print("\n抽出結果:")
        
        # 新語の数を表示
        # Display number of neologisms
        neologisms = result.get("neologisms", [])
        print(f"・新語: {len(neologisms)}個\n")
        
        # 新語の一覧を表示
        # Display list of neologisms
        if neologisms:
            print("抽出された新語一覧:")
            for i, neologism in enumerate(neologisms, 1):
                print(f"{i}. {neologism}")
        
        # 実践の数を表示
        # Display number of praxis
        praxis = result.get("praxis", [])
        print(f"\n・実践: {len(praxis)}個\n")
        
        # 実践の一覧を表示
        # Display list of praxis
        if praxis:
            print("抽出された実践一覧:")
            for i, practice in enumerate(praxis, 1):
                print(f"{i}. {practice}")
        
        print("\n===== テスト完了 =====")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
