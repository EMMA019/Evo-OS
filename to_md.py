from datetime import datetime
import os


def merge_code_to_md(
        output_filename="project_code_summary.md", target_dir="."):
    """
    指定したディレクトリ以下のコードファイルを走査し、1つのMarkdownファイルにまとめます。
    """

    # 修正ポイント: キャッシュフォルダ(.vite, .cache)を追加し、
    # 'node_modules' も明示的にトップレベルで無視。
    IGNORE_DIRS = {
        ".git", "__pycache__", "venv", ".venv", "env", ".env", "node_modules",
        ".idea", ".vscode", "dist", "build", "coverage", "screenshots",
        "__MACOSX", ".vite", ".cache", "temp", "tmp", "target",
        "logs", "assets"    
    }

    # 読み込むファイルの拡張子
    TARGET_EXTENSIONS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".vue",
        ".html", ".css", ".json", ".md", ".txt", ".yml", ".yaml",
        ".gitignore", ".sh", ".bat", ".env", ".env.example",  # Added env files
        ".dockerfile", "makefile", "docker-compose.yml",  # Added docker related files
        ".csproj", ".sln", ".cs",  # Added C# project files
        ".java", ".gradle",  # Added Java project files
        ".go",  # Added Go files
        ".rb",  # Added Ruby files
    }

    current_script = os.path.basename(__file__)

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(f"# Project Code Summary\n\n")
        outfile.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ディレクトリを走査
        for root, dirs, files in os.walk(target_dir):

            # ディレクトリ名を全てチェックし、無視リストにあるものを除外
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            # Sort files for consistent order
            files.sort()

            for file in files:
                # Skip the output file and the script itself
                if file == output_filename or file == current_script:
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, target_dir)

                # Check if file extension is in target extensions or if it's a
                # commonly included file type
                _, ext = os.path.splitext(file)
                is_target_ext = ext.lower() in TARGET_EXTENSIONS
                is_common_file = file.lower() in {
                    'gitignore',
                    'dockerfile',
                    'makefile',
                    'docker-compose.yml',
                    'env',
                    'env.example'}

                if is_target_ext or is_common_file:
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            content = infile.read()

                            # Use relative path for better readability in
                            # Markdown
                            outfile.write(f"## File: `{relative_path}`\n\n")

                            # Determine language for syntax highlighting
                            lang = ext.lstrip(".").lower()
                            if lang == "vue":
                                lang = "html"  # Vue component often uses html-like syntax
                            elif lang == "jsx":
                                lang = "javascript"
                            elif lang == "tsx":
                                lang = "typescript"
                            elif file.lower() == "gitignore" or file.lower() == ".gitignore":
                                lang = "bash"
                            elif file.lower() == "makefile":
                                lang = "makefile"
                            elif file.lower() == "dockerfile":
                                lang = "dockerfile"
                            elif file.lower() == "docker-compose.yml":
                                lang = "yaml"
                            elif lang in ["yml", "yaml"]:
                                lang = "yaml"
                            elif lang == "cs":
                                lang = "csharp"
                            elif lang == "go":
                                lang = "go"
                            elif lang == "rb":
                                lang = "ruby"
                            elif not lang:
                                lang = "text"  # Default for files without extension

                            outfile.write(f"{lang}\n")
                            outfile.write(content)
                            outfile.write("\n\n\n")

                            print(f"Added: {relative_path}")

                    except Exception as e:
                        print(f"Skipped {relative_path}: {e}")

    print(f"\nCompleted! All relevant code merged into: {output_filename}")


# Add datetime import for the generation timestamp

if __name__ == "__main__":
    # Example usage:
    # merge_code_to_md(output_filename="my_project_summary.md", target_dir="../path/to/your/project")
    merge_code_to_md()