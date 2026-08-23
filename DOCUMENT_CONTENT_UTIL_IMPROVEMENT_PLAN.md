# 文書判定・抽出機能の共通パッケージ化 改善計画

## 1. 目的

`ai_chat_util`に混在している次の2種類の責務を分離する。

1. LLM、プロバイダー設定、MCP、API、実行環境を必要とするオーケストレーション
2. 設定なし、または明示的な引数だけで利用できる決定的な文書処理

決定的な文書処理を小さな共通パッケージ`document_content_util`へ移し、
`ai_chat_util`と`document_search_util`の双方から再利用できる状態を目指す。

## 2. 基本方針

依存方向を一方向にする。

```text
document_content_util
        ↑
ai_chat_util / document_search_util
        ↑
CLI / API / MCP / アプリケーション
```

`document_content_util`から`ai_chat_util`、LLMプロバイダー、MCP、FastAPI、
グローバル設定を参照してはならない。

分類基準は単なる「設定の有無」ではなく、次の責務境界とする。

- `document_content_util`: ファイルを安全かつ構造的に判定・抽出する
- `ai_chat_util`: 抽出済みコンテンツをLLMへ渡し、分析・生成・実行する

## 3. 現状

### 3.1 再利用可能な決定的処理

現在、主に`src/ai_chat_util/util/analyze_file_util/`に存在する。

- MagikaによるMIMEタイプ判定
- chardetによる文字コード判定
- テキスト、HTML、Markdown、JSON、YAML等の読み込み
- PDFテキスト抽出
- DOCXテキスト抽出
- XLSXテキスト抽出
- PPTXテキスト抽出
- Office XML構造解析
- ZIP作成・展開
- Markdown解析
- MarkdownからDOCX、XLSX、PPTXへの変換

### 3.2 `ai_chat_util`に残すべき処理

- Chat／Batch Chatクライアント
- OpenAI、Anthropic、Gemini、LiteLLM等のプロバイダー連携
- プロンプト生成とLLMメッセージ変換
- LLMによる画像、PDF、Office、ログ解析
- トークン数、コスト、リトライ、並列実行管理
- CLI、FastAPI、FastMCPインターフェース
- OfficeからPDFへの変換バックエンド選択
- LibreOffice、PyWin32、UNO等の実行環境設定
- SMB、許可ルート、認証を含むファイルサーバー
- Browser Use／Playwright等のエージェント実行

### 3.3 現状の課題

- ファイル判定とLLM解析の責務が同じパッケージに混在している
- `FileUtilDocument.from_file()`がファイル全体をbytesへ読み込む
- Magikaのlabel、score、group等を呼び出し側へ十分に返していない
- 形式別抽出結果が平文中心で、見出し、ページ、表、シート等の構造を統一的に保持しない
- 抽出器、変換器、LLM向けメッセージ生成が密結合している
- `ai_chat_util`全体を依存すると、文書抽出だけの利用者にもLLM、AWS、Browser等の依存が入る
- MarkItDownは未導入

## 4. 新規パッケージ

### 4.1 名称

- 配布名: `document-content-util`
- import名: `document_content_util`

### 4.2 推奨構成

```text
document-content-util/
├─ pyproject.toml
├─ README.md
├─ src/document_content_util/
│  ├─ __init__.py
│  ├─ detection/
│  │  ├─ base.py
│  │  └─ magika_detector.py
│  ├─ extraction/
│  │  ├─ base.py
│  │  ├─ registry.py
│  │  ├─ plain_text.py
│  │  └─ markitdown_extractor.py
│  ├─ fragments/
│  │  └─ models.py
│  ├─ archive/
│  │  └─ zip_handler.py
│  ├─ models.py
│  └─ errors.py
└─ tests/
```

## 5. 公開モデル

最低限、次のモデルを定義する。

```python
class DetectedContentType(BaseModel):
    label: str
    mime_type: str
    group: str
    extensions: list[str]
    is_text: bool
    score: float
    encoding: str | None = None
    extension_matches: bool | None = None


class SourcePointer(BaseModel):
    page_number: int | None = None
    heading_path: list[str] = []
    sheet_name: str | None = None
    slide_number: int | None = None
    table_index: int | None = None
    start_char: int | None = None
    end_char: int | None = None


class DocumentFragment(BaseModel):
    fragment_id: str
    text: str
    pointer: SourcePointer
    metadata: dict[str, object] = {}


class ExtractedDocument(BaseModel):
    source_path: str
    detected: DetectedContentType
    source_media_type: str
    source_checksum: str
    extracted_checksum: str
    text: str
    fragments: list[DocumentFragment]
    detector: str
    detector_version: str
    extractor: str
    extractor_version: str
```

実装時にはPydanticのmutable defaultを`Field(default_factory=...)`へ変更する。

抽出結果には検出結果全体を`detected`として保持する。`source_media_type`は互換性と検索側の簡易参照のために残すが、label、group、score、encoding、拡張子一致情報の正本は`detected`とする。

抽出処理へ適用する安全性と資源制限は、次のポリシーとして明示的に受け取る。

```python
class ExtractionPolicy(BaseModel):
    allowed_roots: list[Path] = Field(default_factory=list)
    max_file_size: int = 100 * 1024 * 1024
    follow_symlinks: bool = False
    max_archive_entries: int = 10_000
    max_archive_uncompressed_size: int = 1 * 1024 * 1024 * 1024
    max_archive_nesting_depth: int = 2
    allow_unknown_binary: bool = False
    allow_executable: bool = False
    enable_markitdown_plugins: bool = False
```

`allowed_roots`が空の場合の扱い、symlinkの拒否または解決方針、TOCTOU対策、サイズ超過時の動作をPhase 0で固定する。パス検証は`extract_local()`だけでなく検出、抽出、アーカイブ展開の各入口で適用する。

## 6. 公開インターフェース

```python
class ContentTypeDetector(Protocol):
    def detect_path(self, path: Path, policy: ExtractionPolicy | None = None) -> DetectedContentType: ...
    def detect_stream(self, stream: BinaryIO) -> DetectedContentType: ...


class DocumentContentExtractor(Protocol):
    def supports(self, detected: DetectedContentType) -> bool: ...
    def extract(self, path: Path, detected: DetectedContentType, policy: ExtractionPolicy | None = None) -> ExtractedDocument: ...
```

利用者向けには薄いFacadeを用意する。

```python
extractor = DocumentExtractor.default()
result = extractor.extract_local(Path("document.docx"), policy=ExtractionPolicy())
```

URL取得は行わず、公開APIはローカルパスまたは明示的なstreamだけを受け付ける。

stream入力では、seek可能なbinary streamだけを受け付ける。`source_path`が存在しない場合の識別子、source checksumの計算方法、読み取り位置の復元を契約化する。パス入力では検出後に同じファイルを再検証し、検出対象と抽出対象の差し替えを検知する。

## 7. 形式判定

Magikaを標準検出器として使用する。

保存する情報:

- label
- MIME type
- group
- extensions
- is_text
- score
- Magikaのバージョン
- 拡張子と検出形式が一致するか

要件:

- ファイル全体をメモリへ読み込まず`identify_path()`またはstream APIを使用する
- 不明バイナリと実行可能形式は既定で拒否する
- 許可形式は呼び出し側がallowlistとして指定できる
- 拡張子はルーティングの主根拠にしない
- 検出結果に拡張子一致フラグを保存するが、不一致だけで直ちに拒否するかはallowlistとポリシーで決める

## 8. 文書抽出

### 8.1 MarkItDown

Office、PDF、HTML等の簡易Markdown変換にMarkItDownを利用する。

要件:

- `convert()`ではなくローカル専用の`convert_local()`を使用する
- URL、HTTP、YouTube等の取得機能は利用しない
- 見出し、表、リスト、リンク等のMarkdown構造を可能な限り保持する
- 変換結果は原本ではなく派生成果物として扱う
- 原本と変換結果のSHA-256を別々に保存する

### 8.2 プレーンテキスト

- Magikaの`is_text`と検出encodingを使用する
- UTF-8固定にしない
- 不正文字の無条件な`errors="ignore"`は避ける
- 変換不能箇所をエラーまたは警告として返す

### 8.3 OCR

初期リリースでは必須にしない。

- スキャンPDFや画像主体文書は`ExtractionUnsupported`または`OCRRequired`として返す
- 将来、OCR実装をpluginとして追加できるインターフェースを用意する
- LLM VisionやクラウドOCRをコア依存にしない

### 8.4 暗号化文書

- パスワードの自動推測・解除を行わない
- 暗号化PDF、Office、ZIPは専用エラーで通知する
- パスワードを引数で受け取る機能はoptionalとし、ログへ出力しない

### 8.5 抽出結果と失敗契約

形式別抽出器が保証するfragmentの粒度を定義する。汎用MarkItDown変換だけでは原本位置を保証できない場合があるため、保証できないpointerのフィールドは`None`とし、推測した位置を返さない。

- Plain text: 文書全体または行単位。文字範囲は正規化後テキスト上の位置とする
- Markdown/HTML: 見出し階層を取得できる場合に`heading_path`へ格納する
- PDF: ページ番号は抽出器が保証できる場合だけ格納する
- XLSX: シート名と、表として認識できる場合のtable indexを格納する
- PPTX: スライド番号を格納する
- DOCX: 見出し・表を保持できる専用抽出器でのみ対応する

抽出失敗は次の例外分類を公開する。

- `ExtractionUnsupported`: 対応する抽出器がない
- `OCRRequired`: 画像主体でOCRが必要
- `EncryptedDocument`: パスワードが必要
- `InvalidDocument`: 破損または形式不正
- `ResourceLimitExceeded`: ファイル、展開後サイズ、件数、ネスト深度の制限超過

単一ファイルAPIは既定で例外を返し、ディレクトリ登録側は例外をファイル単位の失敗レコードへ変換して処理を継続する。失敗レコードにはパス、検出結果が得られた場合のprovenance、エラーコード、秘密情報を含まないメッセージを保存する。

## 9. Optional dependencies

基本依存を小さく保つ。

```toml
[project]
dependencies = [
    "pydantic",
    "magika",
]

[project.optional-dependencies]
office = [
    "markitdown[pdf,docx,pptx,xlsx]",
]
archive = [
    "pyzipper",
]
all = [
    "document-content-util[office,archive]",
]
```

実際のself-reference extraが利用するビルドツールで正しく解決できるかは実装時に検証する。
解決できない場合は`all`へ個別依存を列挙する。

`markitdown`未導入時もパッケージ自体はimportできるようにし、MarkItDownが必要な形式だけ`ExtractionUnsupported`を返す。`DocumentExtractor.default()`はplain textと検出器を利用可能な範囲で構成し、optional依存の有無でimport時に失敗させない。MarkItDown pluginは明示的なポリシー指定がない限り無効とする。

## 10. `ai_chat_util`側の移行

### 10.1 互換アダプター

既存importを直ちに破壊しない。

```python
# ai_chat_util.util.analyze_file_util.document_text_util
from document_content_util import DocumentExtractor

class DocumentTextUtil:
    @classmethod
    def extract_text_from_path(cls, path):
        return DocumentExtractor.default().extract_local(path).text
```

対象:

- `DocumentTextUtil`
- `FileUtilDocument`の形式判定
- `FileUtil.extract_text_from_file_async`
- Word、Excel、PowerPoint、PDFの低レベル抽出ラッパー

`FileUtilDocument`は、`core.analysis.model`での形式判定とLLM向けbytes保持が残るため、Phase 3の初期では削除しない。まず判定処理を`document_content_util`へ委譲し、画像送信、PDFのマルチモーダル入力、base64入力などbytesが必要な経路は既存互換アダプターで維持する。その後、パスベースの抽出とLLM向けbytes入力を別APIとして整理し、利用箇所を確認してから旧モデルを非推奨化する。

### 10.2 `ai_chat_util`に残すFacade

- LLMメッセージへの変換
- Office→PDFバックエンド選択
- 画像を含むマルチモーダル入力生成
- LLMによる要約・分析
- API／MCP／CLIの公開関数

### 10.3 依存削減

移行後、`ai_chat_util`の直接依存から次を削除できるか確認する。

- magika
- python-docx
- python-pptx
- openpyxl
- pdfminer.six
- beautifulsoup4
- chardet
- pyzipper

これらは`document-content-util`の基本依存またはextraへ移す。
`ai_chat_util`がMarkdown→Office生成やOffice→PDF変換で直接使用する依存は残す。

## 11. `document_search_util`側の利用

現在のUTF-8固定`read_text()`を次へ置き換える。

```text
ローカルファイル
  → Magika形式判定
  → allowlist検証
  → MarkItDown／PlainText抽出
  → 構造付きfragment
  → 既存のchunk、Embedding、全軸評価
```

保存するprovenance:

- detected type、MIME、score
- detectorとversion
- extractorとversion
- source checksum
- extracted checksum
- 原本ポインタ
- 変換警告

1ファイルの抽出失敗でディレクトリ全体を停止させず、ファイル単位の結果として返す。

- allowlist、`ExtractionPolicy`、検出結果、抽出警告、source/extracted checksumを既存metadataへ保存する。
- 空文書と抽出失敗を区別し、失敗ファイルを成功ファイルのchunk処理へ混入させない。

## 12. Markdown→Office機能

MarkdownからDOCX、XLSX、PPTXを生成する機能は文書抽出と逆方向である。
初期移行では`ai_chat_util`に残す。

機能が増え、他プロジェクトから再利用される段階で、別パッケージを検討する。

- 配布名候補: `markdown-office-util`
- import名候補: `markdown_office_util`

小規模なうちはパッケージを増やさず、早期分離しない。

## 13. 外出ししない小規模ユーティリティ

次を単独パッケージ化しない。

- logging設定
- request header context
- 単純なpath helper
- downloader単体
- Excel helper単体
- 小さなPydanticモデルだけのパッケージ

これらは責務を所有する主要パッケージの内部モジュールとする。

## 14. 移行フェーズ

### Phase 0: 契約固定

- 現行の対応形式、正常系、失敗系をテスト化
- PDF、DOCX、XLSX、PPTX、UTF-8、非UTF-8、破損、暗号化のfixtureを用意
- 現行公開importと戻り値を一覧化
- `ExtractionPolicy`、失敗レコード、例外分類、fragment pointerの保証範囲を契約テストで固定
- `source_path`、stream識別子、source checksum、extracted checksumの計算規則を固定
- symlink、allowed root、最大サイズ、TOCTOU、ZIP制限の期待動作をfixture化

### Phase 1: パッケージ骨格

- `document-content-util`を作成
- モデル、Protocol、例外、Magika検出器を実装
- ローカル専用とallowlistをテスト
- `markitdown`未導入時のimportとunsupported動作をテスト

### Phase 2: 抽出器

- PlainTextExtractorを実装
- MarkItDownExtractorを実装
- ExtractorRegistryとFacadeを実装
- provenanceとchecksumをテスト

### Phase 3: `ai_chat_util`移行

- 互換アダプターを導入
- 既存API／MCP／CLIテストを維持
- 重複した抽出実装を段階的に削除
- 依存を整理
- `core.analysis.model.FileUtilDocument`とLLM向けbytes経路の互換テストを維持

### Phase 4: `document_search_util`統合

- `read_text()`を共通抽出器へ置換
- 構造付きfragmentと既存chunkモデルを接続
- ファイル単位の失敗結果を追加
- Office/PDF実ファイルで登録・検索を検証

### Phase 5: 高精度抽出

- OCR plugin
- 形式別のページ、見出し、表ポインタ強化
- 必要に応じてAzure Document Intelligence等の外部実装をadapterとして追加

## 15. テスト方針

### 単体テスト

- Magika判定結果
- 拡張子偽装
- 非UTF-8テキスト
- 不明バイナリ
- 拡張子偽装と拡張子不一致のallowlist動作
- 形式別抽出
- 空文書
- 破損文書
- 暗号化文書
- checksum再現性
- fragment ID再現性
- URL拒否
- allowed root外、symlink、最大ファイルサイズ、TOCTOU
- ZIP bomb、ZIP内symlink、ネスト制限、展開件数制限
- optional依存未導入時の既定動作

### 契約テスト

- `ai_chat_util`互換アダプター
- `document_search_util`取り込み結果
- 同じ原本から同じsource checksumが生成される
- 変換器version変更時に派生結果の変更を検知できる

### 統合テスト

- DOCX、XLSX、PPTX、PDFを同じディレクトリから登録
- 原本を変更しない場合は再抽出しない
- 1ファイル失敗時も他ファイルを登録できる
- 見出し・表・シート・スライド由来fragmentを検索できる

## 16. セキュリティ要件

- ローカルファイル専用
- URL schemeを拒否
- allowed root外のパスを拒否できる
- symlink追跡方針を明示する
- 最大ファイルサイズを設定できる
- ZIP展開時にパストラバーサルを防ぐ
- ZIP展開後サイズ、件数、ネスト深度を制限する
- 実行可能形式と不明バイナリを既定拒否
- パスワード、秘密情報、抽出本文を不用意にログへ出さない
- MarkItDown pluginを既定無効にする

## 17. バージョニング

`document-content-util`は独立してSemantic Versioningを行う。

破壊的変更:

- 公開モデルのフィールド削除・型変更
- fragment ID計算規則の変更
- 抽出テキストの正規化規則変更
- 例外型の意味変更

非破壊的変更:

- 新形式の抽出器追加
- optional metadataの追加
- 新しい警告コードの追加

抽出結果へ`extractor_version`を保存し、再現性と再抽出判定に利用する。

## 18. 完了条件

- `document_content_util`が`ai_chat_util`なしでimport・利用できる
- ローカルPDF、DOCX、XLSX、PPTX、テキストを抽出できる
- Magikaの判定情報と抽出provenanceを返す
- URLと不明バイナリを既定拒否する
- `ai_chat_util`の既存公開API／MCP／CLIが互換テストを通過する
- `document_search_util`がOffice/PDFを取り込み、検索できる
- 原本と派生Markdownのchecksumを区別して保存する
- ファイル単位の失敗を他ファイルから隔離する
- 抽出ポリシー、失敗レコード、fragment pointerの保証範囲が公開契約として固定されている
- READMEと移行ガイドが用意されている

## 19. 非目標

初期リリースでは次を対象外とする。

- LLMを使った文書要約
- LLM VisionによるOCR
- Office→PDFの実行環境管理
- SMBやHTTPからのファイル取得
- MCP／FastAPIサーバー
- Markdown→Office生成機能の即時分離
- 全形式での高忠実度なレイアウト再現

## 20. 意思決定記録

- 共通パッケージ名は`document_content_util`を第一候補とする
- `ai_chat_util`はLLMオーケストレーション層として維持する
- 形式判定にはMagikaを使用する
- Office/PDFの初期抽出にはMarkItDownを使用する
- MarkItDownはローカル専用APIを使用し、pluginは既定無効とする
- 原本と抽出結果を別のchecksumで管理する
- 構造付きfragmentと原本ポインタを公開契約に含める
- 既存利用者のため`ai_chat_util`に互換アダプターを残す
