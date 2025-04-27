# 改作業助手 Agent 架構設計

本文檔描述了用於自動評分學生作業的 Agent 的架構設計。該 Agent 旨在支援多種大型語言模型 (LLM)，並採用非同步處理以提高效率。

## 1. 核心架構

Agent 採用基於 `asyncio` 的模組化非同步架構。

```mermaid
graph LR
    A[Async Agent Core] -- await --> B(Input Processor);
    A -- await --> C(LLM Interface);
    A -- await --> D(Grading Logic);
    A -- await --> E(Output Handler);
    A --> F(Configuration Manager);
    A --> G(Logger);

    B -- async methods --> B1(Directory Scanner);
    B -- async methods --> B2(File Handler);
    B2 -- async methods --> B3(File Converter);
    B2 -- async methods --> B4(Content Extractor);

    C -- async methods --> C1(Async LLM Client - Abstract);
    C1 -- async methods --> C2(Async OpenAI Client);
    C1 -- async methods --> C3(Async Anthropic Client);
    C1 -- async methods --> C4(Async Gemini Client - Deferred);
    C --> C5(LLM Factory);

    D --> D1(Prompt Generator);
    D --> D2(Score Parser);

    E -- async methods --> E1(Result Formatter);
    E -- async methods --> E2(Async CSV Writer);
    E -- async methods --> E3(... other Writers);

    subgraph Input Processing (Async IO)
        B1
        B2
        B3
        B4
    end

    subgraph LLM Abstraction (Async + Retry)
        C1
        C2
        C3
        C4
        C5
    end

    subgraph Grading
        D1
        D2
    end

    subgraph Output (Async IO)
        E1
        E2
        E3
    end
```

**組件職責:**

*   **Async Agent Core:** 作為非同步事件循環的協調者，使用 `await` 調度其他非同步組件。負責並發處理多個學生的評分任務 (例如使用 `asyncio.gather`)。
*   **Input Processor:** 負責處理輸入資料。所有涉及檔案讀取、掃描和轉換的操作都將使用非同步 I/O (例如 `aiofiles`)。
    *   `Directory Scanner`: 非同步掃描指定的學生作業根目錄。
    *   `File Handler`: 非同步處理每個學生的檔案，判斷檔案類型。
    *   `File Converter`: (可重用 `prepare.py` 的邏輯) 非同步將特殊格式 (ipynb, docx, pptx, pdf) 轉換為 LLM 可理解的格式。
    *   `Content Extractor`: 非同步從原始檔案或轉換後的檔案中提取 Base64 編碼的圖片列表和文字內容。
*   **LLM Interface (抽象層):** 統一與不同 LLM 互動的非同步介面。
    *   `Async LLM Client` (抽象基類/介面): 定義標準的非同步互動方法，例如 `async def grade(prompt, content_list)`。
    *   `Async OpenAI Client`, `Async Anthropic Client` (具體實作): 繼承/實作 `Async LLM Client`，封裝各自非同步 API 的調用細節、認證、輸入格式轉換和輸出解析。**包含重試邏輯** (例如使用 `tenacity` 庫)。
    *   `Async Gemini Client`: 暫緩實作，待後續提供 API 文件後添加。
    *   `LLM Factory`: 根據配置動態創建所需的 `Async LLM Client` 實例。
*   **Grading Logic:** 負責評分的核心邏輯。
    *   `Prompt Generator`: 根據作業要求、學生資訊、提取的作業內容，以及系統提示，動態生成適合目標 LLM 的完整 Prompt。
    *   `Score Parser`: 解析來自 `Async LLM Client` 的回應，提取結構化的評分結果 (學號、分數、評語)，並進行驗證。
*   **Output Handler:** 負責處理和儲存評分結果。
    *   `Result Formatter`: 將結構化的評分結果整理成特定格式。
    *   `Async CSV Writer`: 使用非同步 I/O 將格式化後的結果寫入 CSV 檔案。
*   **Configuration Manager:** 集中管理系統配置。**支援從 YAML 檔案讀取配置** (使用 `PyYAML`)，也可兼容 `.env`。
*   **Logger:** 提供統一的日誌記錄介面，記錄系統運行資訊、警告和錯誤。

## 2. LLM 抽象層設計

*   **通用輸入/輸出資料結構:**
    ```python
    from typing import List, Any
    from dataclasses import dataclass

    @dataclass
    class LLMInput:
        student_id: str
        student_name: str
        text_content: List[str]
        image_content: List[str] # Base64 encoded images

    @dataclass
    class LLMOutput:
        student_id: str
        score: int
        comment: str
        raw_response: Any = None
    ```
*   **抽象客戶端 (`AsyncLLMClient`):**
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any
    import asyncio
    from tenacity import retry, stop_after_attempt, wait_fixed # 引入重試庫

    class AsyncLLMClient(ABC):
        @abstractmethod
        def __init__(self, api_key: str, model_name: str, **kwargs):
            """初始化客戶端"""
            pass

        @abstractmethod
        async def _prepare_request_payload(self, system_prompt: str, requirements: str, llm_input: LLMInput) -> Dict[str, Any]:
            """(非同步) 將標準輸入轉換為特定 LLM API 的請求格式"""
            pass

        @abstractmethod
        async def _parse_response(self, response: Any, student_id: str) -> LLMOutput:
            """(非同步) 解析特定 LLM API 的回應為標準輸出格式"""
            pass

        # 加入重試機制，例如重試3次，每次間隔2秒
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        @abstractmethod
        async def grade(self, system_prompt: str, requirements: str, llm_input: LLMInput) -> LLMOutput:
            """(非同步) 執行評分請求並返回標準化結果"""
            # 1. await _prepare_request_payload
            # 2. await 發送 API 請求 (使用對應庫的 async client)
            # 3. await _parse_response
            # 4. 返回 LLMOutput
            pass
    ```

## 3. 重構步驟規劃

1.  **建立專案結構:** 創建 `grading_agent/` 目錄及子目錄 (`core`, `processors`, `llm_interface`, `grading`, `output`, `config`, `utils`)。
2.  **定義核心抽象:** 定義 `AsyncLLMClient`, `LLMInput`, `LLMOutput`。
3.  **實作 Async LLM 客戶端:** 封裝 OpenAI 和 Anthropic 邏輯到 `AsyncOpenAIClient`, `AsyncAnthropicClient`，使用非同步 API 並加入重試。
4.  **遷移檔案處理 (Async):** 重構 `read_files`, `process_file`, `prepare.py` 到 `Input Processor`，使用 `aiofiles`。
5.  **遷移評分邏輯:** 將 Prompt 生成和回應解析移至 `Grading Logic`。
6.  **實作輸出處理 (Async):** 將 CSV 寫入移至 `Output Handler` (`AsyncCSVWriter`)，使用非同步 I/O。
7.  **實作配置管理 (YAML):** 引入 `Configuration Manager`，添加讀取 YAML 功能。
8.  **構建 Async Agent Core:** 編寫 `Async Agent Core`，使用 `asyncio` 協調流程。
9.  **引入日誌:** 使用標準 `logging`。
10. **測試:** 編寫非同步代碼的單元和整合測試。
11. **(暫緩) 添加 Gemini 支援。**