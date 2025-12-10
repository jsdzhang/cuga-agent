from dotenv import load_dotenv
from e2b import Template, default_build_logger

load_dotenv()

template = (
    Template()
    .from_template("code-interpreter-v1")
    .pip_install(['langchain'])  # Install Python packages
    .npm_install(['langchain'])  # Install Node.js packages
)

if __name__ == '__main__':
    Template.build(
        template,
        alias="cuga-langchain",
        cpu_count=2,
        memory_mb=2048,
        on_build_logs=default_build_logger(),
    )
