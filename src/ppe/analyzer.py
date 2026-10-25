"""Сопоставление касок и жилетов с обнаруженными людьми."""

from src.inference.types import Detection


def center_in_region(
    item: Detection,
    x1: int,
    y1: float,
    x2: int,
    y2: float,
) -> bool:
    """Проверить, находится ли центр предмета внутри области."""
    center_x = (item.x1 + item.x2) / 2
    center_y = (item.y1 + item.y2) / 2

    return (
        x1 <= center_x <= x2
        and y1 <= center_y <= y2
    )


def check_ppe(people: list[Detection], equipment: list[Detection], min_person_height: int, frame_height: int) -> list[dict]:
    results = []

    for person in people:
        if person.class_name != "person" or person.track_id is None:
            continue

        height = person.y2 - person.y1

        # Мелкого или обрезанного сверху человека не оцениваем.
        if height < min_person_height or person.y1 <= 0:
            continue

        has_helmet = any(
            item.class_name == "helmet"
            and center_in_region(
                item,
                person.x1,
                person.y1,
                person.x2,
                person.y1 + height * 0.35,
            )
            for item in equipment
        )

        # Если кадр обрезан выше нижней границы туловища,
        # наличие жилета оценивать не будем.
        torso_bottom = person.y1 + height * 0.75

        has_vest = None

        if torso_bottom < frame_height - 1:
            has_vest = any(
                item.class_name == "vest"
                and center_in_region(
                    item,
                    person.x1,
                    person.y1 + height * 0.20,
                    person.x2,
                    torso_bottom,
                )
                for item in equipment
            )

        results.append({
            "person": person,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
        })

    return results