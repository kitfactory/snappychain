import os
import datetime
from typing import List, Optional
from snappychain.print import verbose_print, debug_print, Color
import wikipedia

def _format_wiki_content(content: str) -> str:
    """
    Format Wikipedia content into structured markdown.
    Wikipedia本文を構造化されたマークダウンに整形します。

    Args:
        content (str): Raw Wikipedia content
                      生のWikipedia本文

    Returns:
        str: Formatted markdown content
             整形されたマークダウン本文
    """
    lines = content.split('\n')
    formatted_lines = []
    in_list = False
    in_quote = False
    last_was_header = False
    prev_was_header = False
    
    for line in lines:
        # Remove excessive spaces
        # 過剰なスペースを削除
        line = line.strip()
        if not line:
            if not last_was_header:
                formatted_lines.append('')
            in_list = False
            in_quote = False
            continue

        # Handle section headers (== Title ==)
        # セクション見出しの処理（== タイトル ==）
        if line.startswith('==') and line.endswith('=='):
            # Add newline before header only if previous line was not a header and not empty
            # 前の行が見出しでなく、空行でもない場合のみ改行を追加
            if formatted_lines and not prev_was_header and formatted_lines[-1] != '':
                formatted_lines.append('')
            level = line.count('=') // 2
            title = line.strip('= ')
            formatted_lines.append(f"{'#' * (level + 1)} {title}")
            last_was_header = True
            prev_was_header = True
            continue

        # Handle bullet points
        # 箇条書きの処理
        if line.startswith('*') or line.startswith('•'):
            formatted_lines.append(f"- {line[1:].strip()}")
            in_list = True
            last_was_header = False
            prev_was_header = False
            continue

        # Handle numbered lists
        # 番号付きリストの処理
        if line[0].isdigit() and line[1:].startswith('. '):
            formatted_lines.append(line)
            in_list = True
            last_was_header = False
            prev_was_header = False
            continue

        # Handle quotes
        # 引用の処理
        if line.startswith('>') or line.startswith('"'):
            if not in_quote:
                formatted_lines.append('')
                in_quote = True
            line = line.lstrip('>"')
            formatted_lines.append(f"> {line}")
            last_was_header = False
            prev_was_header = False
            continue

        # Handle regular paragraphs
        # 通常の段落の処理
        if not in_list and not in_quote:
            formatted_lines.append(line)
            last_was_header = False
            prev_was_header = False

    # Remove any trailing empty lines
    # 末尾の空行を削除
    while formatted_lines and not formatted_lines[-1]:
        formatted_lines.pop()

    return '\n'.join(formatted_lines)


def wikipedia_to_text(items: list[str], dest: str):
    """
    Download Wikipedia pages and save each page to a separate markdown file in the specified directory.
    Wikipediaページをダウンロードし、指定されたディレクトリ内に各ページを個別のマークダウンファイルとして保存します。

    Args:
        items (list[str]): List of Wikipedia page titles
                          Wikipediaページタイトルのリスト
        dest (str): Destination directory path
                   保存先ディレクトリパス
    """
    # Set Wikipedia language to Japanese
    # Wikipediaの言語を日本語に設定
    wikipedia.set_lang("ja")

    try:
        # Create destination directory if it doesn't exist
        # 保存先ディレクトリが存在しない場合は作成
        os.makedirs(dest, exist_ok=True)

        # Process each Wikipedia page
        # 各Wikipediaページを処理
        for item in items:
            try:
                # Create a safe filename from the page title
                # ページタイトルから安全なファイル名を作成
                safe_filename = "".join(c for c in item if c.isalnum() or c in (' ', '-', '_')).rstrip()
                file_path = os.path.join(dest, f"{safe_filename}.md")

                # Get Wikipedia page
                # Wikipediaページを取得
                verbose_print("Wikipedia", f"Downloading page: {item}", Color.YELLOW)
                page = wikipedia.page(item)
                
                # Write page content to individual markdown file
                # ページ内容を個別のマークダウンファイルに書き込み
                with open(file_path, "w", encoding="utf-8") as f:
                    # Write title as H1 header
                    # タイトルをH1ヘッダーとして書き込み
                    f.write(f"# {page.title}\n\n")
                    
                    # Write metadata
                    # メタデータを書き込み
                    f.write("## メタデータ\n\n")
                    f.write(f"- 取得日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- ソース: {page.url}\n")
                    f.write(f"- カテゴリ: {', '.join(page.categories)}\n")
                    f.write(f"- 要約: {page.summary}\n\n")
                    
                    # Write formatted content
                    # 整形された本文を書き込み
                    f.write("## 本文\n\n")
                    f.write(_format_wiki_content(page.content))
                    f.write("\n")
                
                verbose_print("Wikipedia", f"Downloaded page to: {file_path}", Color.GREEN)
                debug_print("Wikipedia", f"Page details: {len(page.content)} characters, {len(page.categories)} categories", Color.CYAN)
            
            except Exception as e:
                verbose_print("Wikipedia", f"Error downloading page '{item}': {str(e)}", Color.RED)
                continue

        verbose_print("Wikipedia", f"Completed writing all pages to {dest}", Color.GREEN)

    except Exception as e:
        verbose_print("Wikipedia", f"Error creating directory '{dest}': {str(e)}", Color.RED)
        raise


if __name__ == "__main__":
    wikipedia_to_text(
        items=["アオダモ", "アオハダ", "イロハモミジ","ヤマボウシ","カツラ","ソヨゴ","ツリバナ","ハイノキ"],
        dest="examples/documents/tees"
    )