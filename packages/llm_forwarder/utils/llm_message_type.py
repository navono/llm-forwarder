from pydantic import BaseModel, Field

HTTP_OK = 200


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: list[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: int | None = Field(None, description="最大生成token数")
    temperature: float | None = Field(1.0, description="温度参数")
    top_p: float | None = Field(1.0, description="top_p参数")
    top_k: int | None = Field(0, description="top_k参数")
    stream: bool | None = Field(False, description="是否流式输出")
    stop: str | list[str] | None = Field(None, description="停止词")
    frequency_penalty: float | None = Field(0.0, description="频率惩罚参数")
    presence_penalty: float | None = Field(0.0, description="存在惩罚参数")
    repetition_penalty: float | None = Field(1.0, description="重复惩罚参数")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class AudioSpeechRequest(BaseModel):
    model: str = Field(..., description="TTS模型名称")
    input: str = Field(..., description="要转换的文本")
    voice: str = Field("alloy", description="语音类型")
    response_format: str | None = Field("mp3", description="音频格式")
    speed: float | None = Field(1.0, description="语音速度")


class AudioTranscriptionResponse(BaseModel):
    text: str


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="嵌入模型名称")
    input: str | list[str] = Field(..., description="输入文本")
    # encoding_format: Optional[str] = Field("float", description="编码格式")
    task: str | None = Field("retrieval.passage", description="下游任务类型，默认为 text-matching")
    dimensions: int | None = Field(1024, description="嵌入维度，默认为 1024")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class RerankRequest(BaseModel):
    model: str = Field(..., description="重排序模型名称")
    query: str = Field(..., description="查询文本")
    documents: list[str] = Field(..., description="文档列表")
    top_k: int | None = Field(None, description="返回前k个结果")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str


class RerankResponse(BaseModel):
    object: str = "list"
    data: list[RerankResult]
    model: str
    usage: dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    capabilities: dict = Field(default_factory=dict, description="模型能力")


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class DeepSearchRequest(BaseModel):
    model: str = Field(..., description="深度搜索模型名称")
    messages: list[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: int | None = Field(None, description="最大生成token数")
    temperature: float | None = Field(1.0, description="温度参数")
    search_depth: int | None = Field(3, description="搜索深度")
    include_sources: bool | None = Field(True, description="是否包含来源")


class ClassifyRequest(BaseModel):
    model: str = Field(..., description="分类模型名称")
    input: str | list[str] = Field(..., description="输入文本")
    labels: list[str] = Field(..., description="分类标签列表")
    multi_label: bool | None = Field(False, description="是否多标签分类")


class ClassifyResult(BaseModel):
    label: str
    score: float
    index: int


class ClassifyResponse(BaseModel):
    object: str = "list"
    data: list[list[ClassifyResult]]  # 每个输入对应一个分类结果列表
    model: str
    usage: dict[str, int]


class SearchRequest(BaseModel):
    model: str = Field(..., description="搜索模型名称")
    query: str = Field(..., description="搜索查询")
    top_k: int | None = Field(10, description="返回结果数量")
    return_json: bool | None = Field(False, description="是否返回 JSON 格式")
    summary: bool | None = Field(False, description="是否使用 LLM 总结搜索结果")
    stream: bool | None = Field(False, description="是否使用流式返回搜索结果")


class SearchResult(BaseModel):
    title: str
    url: str
    description: str
    date: str | None = None


class SearchResponse(BaseModel):
    object: str = "list"
    data: list[SearchResult]
    model: str
