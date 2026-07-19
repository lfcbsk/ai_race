from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedDocument:
    """
    Kết quả sau khi chuẩn hóa một văn bản.

    normalized_to_raw[i] cho biết ký tự thứ i trong
    normalized_text được tạo từ vị trí nào trong raw_text.

    Với span normalized [start, end), có thể ánh xạ về raw bằng:
        raw_start = normalized_to_raw[start]
        raw_end = normalized_to_raw[end - 1] + 1
    """

    raw_text: str
    normalized_text: str
    normalized_to_raw: list[int]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if len(self.normalized_text) != len(
            self.normalized_to_raw
        ):
            raise ValueError(
                "Độ dài normalized_text phải bằng "
                "độ dài normalized_to_raw."
            )

    def normalized_span_to_raw(
        self,
        start: int,
        end: int,
    ) -> tuple[int, int]:
        """
        Chuyển span [start, end) trên normalized_text
        về span [raw_start, raw_end) trên raw_text.
        """
        if not 0 <= start < end <= len(
            self.normalized_text
        ):
            raise ValueError(
                f"Normalized span không hợp lệ: "
                f"[{start}, {end})"
            )

        raw_start = self.normalized_to_raw[start]
        raw_end = (
            self.normalized_to_raw[end - 1] + 1
        )

        return raw_start, raw_end

    def raw_span_to_normalized(
        self,
        raw_start: int,
        raw_end: int,
    ) -> tuple[int, int] | None:
        """
        Chuyển span raw [raw_start, raw_end)
        sang normalized text.

        Trả None nếu toàn bộ raw span đã bị loại bỏ
        trong quá trình normalize.
        """
        if not 0 <= raw_start < raw_end <= len(
            self.raw_text
        ):
            raise ValueError(
                f"Raw span không hợp lệ: "
                f"[{raw_start}, {raw_end})"
            )

        matched_positions = [
            normalized_index
            for normalized_index, raw_index
            in enumerate(self.normalized_to_raw)
            if raw_start <= raw_index < raw_end
        ]

        if not matched_positions:
            return None

        normalized_start = matched_positions[0]
        normalized_end = matched_positions[-1] + 1

        return normalized_start, normalized_end


# Các ký tự điều khiển được phép giữ lại.
ALLOWED_CONTROL_CHARACTERS = {
    "\n",
    "\t",
}


# Chỉ chuẩn hóa các ký tự có ý nghĩa tương đương rõ ràng.
CHARACTER_REPLACEMENTS = {
    # Non-breaking space
    "\u00a0": " ",

    # Zero-width characters
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",

    # Dấu nháy
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",

    # Gạch ngang
    "‐": "-",
    "-": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",

    # Dấu chấm ba
    "…": "...",

    # Dấu nhân
    "×": "x",

    # Slash full-width
    "／": "/",

    # Colon full-width
    "：": ":",

    # Parentheses full-width
    "（": "(",
    "）": ")",

    # Comma and period full-width
    "，": ",",
    "．": ".",
}


@dataclass
class _CharacterBuffer:
    """
    Buffer nội bộ để xây normalized text cùng offset map.
    """

    characters: list[str] = field(
        default_factory=list
    )

    raw_positions: list[int] = field(
        default_factory=list
    )

    def append(
        self,
        character: str,
        raw_position: int,
    ) -> None:
        """
        Thêm một hoặc nhiều ký tự được tạo từ cùng
        một vị trí raw.
        """
        for output_character in character:
            self.characters.append(output_character)
            self.raw_positions.append(raw_position)

    def pop(self) -> None:
        if self.characters:
            self.characters.pop()
            self.raw_positions.pop()

    def last_character(self) -> str | None:
        if not self.characters:
            return None

        return self.characters[-1]

    def build(self) -> tuple[str, list[int]]:
        return (
            "".join(self.characters),
            self.raw_positions,
        )


def _normalize_unicode_clusters(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Chuẩn hóa Unicode NFC theo từng cụm base character + combining marks.

    Normalize từng code point riêng lẻ không thể ghép các chuỗi decomposed,
    ví dụ ``a`` + combining acute accent. Mọi ký tự đầu ra của một cụm
    được map về vị trí raw đầu tiên đã tạo ra cụm đó.
    """
    if not text:
        return "", []

    buffer = _CharacterBuffer()
    cluster_characters: list[str] = []
    cluster_positions: list[int] = []

    def flush_cluster() -> None:
        if not cluster_characters:
            return
        normalized_cluster = unicodedata.normalize(
            "NFC", "".join(cluster_characters)
        )
        first_raw_position = cluster_positions[0]
        buffer.append(normalized_cluster, first_raw_position)
        cluster_characters.clear()
        cluster_positions.clear()

    for character, raw_position in zip(text, text_to_raw):
        if cluster_characters and unicodedata.combining(character) == 0:
            flush_cluster()
        cluster_characters.append(character)
        cluster_positions.append(raw_position)

    flush_cluster()
    return buffer.build()


def _is_disallowed_control_character(
    character: str,
) -> bool:
    """
    Kiểm tra ký tự control không nên tồn tại
    trong dữ liệu clinical text.
    """
    if character in ALLOWED_CONTROL_CHARACTERS:
        return False

    category = unicodedata.category(character)

    return category in {
        "Cc",
        "Cf",
    }


def _normalize_newlines(
    raw_text: str,
) -> tuple[str, list[int]]:
    """
    Chuẩn hóa:
        \\r\\n -> \\n
        \\r   -> \\n

    Đồng thời tạo map từ text trung gian về raw text.
    """
    buffer = _CharacterBuffer()

    index = 0

    while index < len(raw_text):
        character = raw_text[index]

        if character == "\r":
            buffer.append(
                "\n",
                raw_position=index,
            )

            if (
                index + 1 < len(raw_text)
                and raw_text[index + 1] == "\n"
            ):
                index += 2
            else:
                index += 1

            continue

        buffer.append(
            character,
            raw_position=index,
        )

        index += 1

    return buffer.build()


def _apply_character_replacements(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Thay các Unicode punctuation variant bằng
    ký tự thông thường nhưng giữ mapping về raw.
    """
    buffer = _CharacterBuffer()

    for index, character in enumerate(text):
        raw_position = text_to_raw[index]

        if _is_disallowed_control_character(
            character
        ):
            continue

        replacement = CHARACTER_REPLACEMENTS.get(
            character,
            character,
        )

        if not replacement:
            continue

        buffer.append(
            replacement,
            raw_position=raw_position,
        )

    replaced_text, replaced_mapping = buffer.build()
    return _normalize_unicode_clusters(
        replaced_text,
        replaced_mapping,
    )


def _collapse_horizontal_whitespace(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Chuẩn hóa chuỗi space/tab liên tiếp thành một space.

    Không collapse newline để giữ cấu trúc hồ sơ.
    """
    buffer = _CharacterBuffer()
    in_horizontal_whitespace = False

    for index, character in enumerate(text):
        raw_position = text_to_raw[index]

        if character in {
            " ",
            "\t",
            "\v",
            "\f",
        }:
            if not in_horizontal_whitespace:
                buffer.append(
                    " ",
                    raw_position=raw_position,
                )

            in_horizontal_whitespace = True
            continue

        in_horizontal_whitespace = False

        buffer.append(
            character,
            raw_position=raw_position,
        )

    return buffer.build()


def _remove_spaces_around_newlines(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Xóa space ở cuối dòng và đầu dòng.

    Ví dụ:
        "abc   \\n   def"
    thành:
        "abc\\ndef"
    """
    buffer = _CharacterBuffer()

    for index, character in enumerate(text):
        raw_position = text_to_raw[index]

        if character == "\n":
            while buffer.last_character() == " ":
                buffer.pop()

            buffer.append(
                "\n",
                raw_position=raw_position,
            )
            continue

        if (
            character == " "
            and buffer.last_character() == "\n"
        ):
            continue

        buffer.append(
            character,
            raw_position=raw_position,
        )

    return buffer.build()


def _collapse_blank_lines(
    text: str,
    text_to_raw: list[int],
    max_consecutive_newlines: int = 2,
) -> tuple[str, list[int]]:
    """
    Giới hạn số newline liên tiếp.

    max_consecutive_newlines=2 nghĩa là giữ tối đa
    một dòng trắng giữa hai đoạn.
    """
    buffer = _CharacterBuffer()
    consecutive_newlines = 0

    for index, character in enumerate(text):
        raw_position = text_to_raw[index]

        if character == "\n":
            consecutive_newlines += 1

            if (
                consecutive_newlines
                <= max_consecutive_newlines
            ):
                buffer.append(
                    character,
                    raw_position=raw_position,
                )

            continue

        consecutive_newlines = 0

        buffer.append(
            character,
            raw_position=raw_position,
        )

    return buffer.build()


def _normalize_spaces_before_punctuation(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Xóa space trước dấu câu rõ ràng.

    Ví dụ:
        "đau đầu , sốt ."
    thành:
        "đau đầu, sốt."

    Không áp dụng rộng cho slash, dấu trừ hoặc dấu cộng
    vì chúng thường xuất hiện trong đơn vị và kết quả xét nghiệm.
    """
    punctuation = {
        ",",
        ".",
        ";",
        ":",
        "!",
        "?",
        ")",
        "]",
        "}",
    }

    buffer = _CharacterBuffer()

    for index, character in enumerate(text):
        raw_position = text_to_raw[index]

        if character in punctuation:
            while buffer.last_character() == " ":
                buffer.pop()

        buffer.append(
            character,
            raw_position=raw_position,
        )

    return buffer.build()


def _normalize_spaces_after_opening_bracket(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Xóa space ngay sau dấu ngoặc mở.

    Ví dụ:
        "( đau đầu )"
    thành:
        "(đau đầu)"
    """
    opening_brackets = {
        "(",
        "[",
        "{",
    }

    buffer = _CharacterBuffer()

    for index, character in enumerate(text):
        raw_position = text_to_raw[index]

        if (
            character == " "
            and buffer.last_character()
            in opening_brackets
        ):
            continue

        buffer.append(
            character,
            raw_position=raw_position,
        )

    return buffer.build()


def _strip_document_edges(
    text: str,
    text_to_raw: list[int],
) -> tuple[str, list[int]]:
    """
    Loại bỏ whitespace ở đầu và cuối document.
    """
    start = 0
    end = len(text)

    while (
        start < end
        and text[start].isspace()
    ):
        start += 1

    while (
        end > start
        and text[end - 1].isspace()
    ):
        end -= 1

    return (
        text[start:end],
        text_to_raw[start:end],
    )


def normalize_text(
    raw_text: str,
    *,
    collapse_spaces: bool = True,
    normalize_punctuation: bool = True,
    preserve_line_breaks: bool = True,
    max_consecutive_newlines: int = 2,
    strip_edges: bool = True,
) -> NormalizedDocument:
    """
    Chuẩn hóa clinical text và giữ offset mapping.

    Parameters
    ----------
    raw_text:
        Văn bản gốc.

    collapse_spaces:
        Chuyển nhiều horizontal whitespace thành một space.

    normalize_punctuation:
        Chuẩn hóa dấu câu Unicode và khoảng trắng quanh dấu câu.

    preserve_line_breaks:
        Nếu True, giữ newline để phục vụ section detection.
        Nếu False, newline được chuyển thành space.

    max_consecutive_newlines:
        Số newline liên tiếp tối đa được giữ lại.

    strip_edges:
        Xóa whitespace đầu và cuối document.
    """
    if not isinstance(raw_text, str):
        raise TypeError(
            "raw_text phải là string."
        )

    if not raw_text:
        return NormalizedDocument(
            raw_text=raw_text,
            normalized_text="",
            normalized_to_raw=[],
            metadata={
                "is_empty": True,
            },
        )

    # Bước 1: chuẩn hóa newline và tạo initial mapping.
    text, mapping = _normalize_newlines(
        raw_text
    )

    # Bước 2: xử lý punctuation variant và control character.
    text, mapping = _apply_character_replacements(
        text,
        mapping,
    )

    # Bước 3: có thể biến newline thành space
    # nếu model không cần cấu trúc dòng.
    if not preserve_line_breaks:
        replaced_characters: list[str] = []
        replaced_mapping: list[int] = []

        for character, raw_position in zip(
            text,
            mapping,
        ):
            if character == "\n":
                replaced_characters.append(" ")
            else:
                replaced_characters.append(
                    character
                )

            replaced_mapping.append(raw_position)

        text = "".join(replaced_characters)
        mapping = replaced_mapping

    # Bước 4: collapse horizontal whitespace.
    if collapse_spaces:
        text, mapping = (
            _collapse_horizontal_whitespace(
                text,
                mapping,
            )
        )

    # Bước 5: làm sạch whitespace theo cấu trúc dòng.
    if preserve_line_breaks:
        text, mapping = (
            _remove_spaces_around_newlines(
                text,
                mapping,
            )
        )

        text, mapping = _collapse_blank_lines(
            text,
            mapping,
            max_consecutive_newlines=(
                max_consecutive_newlines
            ),
        )

    # Bước 6: chuẩn hóa spacing quanh punctuation.
    if normalize_punctuation:
        text, mapping = (
            _normalize_spaces_before_punctuation(
                text,
                mapping,
            )
        )

        text, mapping = (
            _normalize_spaces_after_opening_bracket(
                text,
                mapping,
            )
        )

    # Bước 7: trim document.
    if strip_edges:
        text, mapping = _strip_document_edges(
            text,
            mapping,
        )

    result = NormalizedDocument(
        raw_text=raw_text,
        normalized_text=text,
        normalized_to_raw=mapping,
        metadata={
            "raw_length": len(raw_text),
            "normalized_length": len(text),
            "collapse_spaces": collapse_spaces,
            "normalize_punctuation": (
                normalize_punctuation
            ),
            "preserve_line_breaks": (
                preserve_line_breaks
            ),
        },
    )

    validate_normalized_document(result)

    return result


def validate_normalized_document(
    document: NormalizedDocument,
) -> None:
    """
    Kiểm tra consistency của normalized text và offset map.
    """
    if len(document.normalized_text) != len(
        document.normalized_to_raw
    ):
        raise ValueError(
            "normalized_text và normalized_to_raw "
            "không cùng độ dài."
        )

    previous_raw_position = -1

    for normalized_index, raw_position in enumerate(
        document.normalized_to_raw
    ):
        if not isinstance(raw_position, int):
            raise TypeError(
                "Mọi phần tử normalized_to_raw "
                "phải là số nguyên."
            )

        if not 0 <= raw_position < len(
            document.raw_text
        ):
            raise ValueError(
                f"Offset raw không hợp lệ tại "
                f"normalized index {normalized_index}: "
                f"{raw_position}"
            )

        # Mapping phải giữ thứ tự tăng dần,
        # nhưng có thể bằng nhau khi một ký tự raw
        # sinh ra nhiều ký tự normalized, ví dụ … -> ...
        if raw_position < previous_raw_position:
            raise ValueError(
                "normalized_to_raw không giữ thứ tự."
            )

        previous_raw_position = raw_position


def normalize_document(
    document: Any,
    **normalize_kwargs: Any,
) -> Any:
    """
    Chuẩn hóa một MedicalDocument mà không thay raw_text.

    Hàm dùng Any để tránh circular import.
    MedicalDocument nên có:
        - raw_text
        - metadata

    Hàm sẽ thêm:
        - normalized_text
        - normalized_to_raw

    Nếu dataclass MedicalDocument chưa có các field này,
    nên cập nhật schemas.py như phần bên dưới.
    """
    normalized = normalize_text(
        document.raw_text,
        **normalize_kwargs,
    )

    document.normalized_text = (
        normalized.normalized_text
    )

    document.normalized_to_raw = (
        normalized.normalized_to_raw
    )

    document.metadata.update(
        normalized.metadata
    )

    return document
