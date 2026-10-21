"""Offline checks for prompt building: audience detection and developer-detail stripping (no Ollama).

Run: .venv\\Scripts\\python.exe tests\\test_prompt.py   (or with pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.prompt import (AUDIENCE_DEVELOPER, AUDIENCE_USER, TECHNICAL_LEAK, audience,  # noqa: E402
                                   build_messages, strip_developer_details)

CHUNK = ("- `/admin/add-student` (`client/src/pages/admin/AddStudent.jsx`). The admin chooses a class.\n"
         "2. **API.** `POST /api/admin/create-student`. Handler `adminController.createStudent` (line 37).\n"
         "- Used by the mentor pages `AssignStudents.jsx` and `CourseTeacherPortal.jsx`. Body `{studentIds}`.")


def test_audience():
    assert audience("How may an admin add students?") == AUDIENCE_USER
    assert audience("How can I delete a class?") == AUDIENCE_USER  # everyday words stay "user"
    assert audience("I didn't get the reset code") == AUDIENCE_USER
    assert audience("What API endpoint creates a student?") == AUDIENCE_DEVELOPER
    assert audience("What does POST /api/courses/create-course need?") == AUDIENCE_DEVELOPER


def test_strip_keeps_pages_and_removes_developer_details():
    out = strip_developer_details(CHUNK)
    assert "/admin/add-student" in out and "The admin chooses a class." in out
    for gone in ("/api/", ".jsx", "createStudent", "{studentIds}", "line 37", "()"):
        assert gone not in out, gone
    assert not TECHNICAL_LEAK.search(out)
    evidence = "- `server/helpers/teachingScope.js`: `authorizeCourse`, which allows the mentor only.\n- `RUNS_CLASSROOM`, `teacherHasCourse`\n"
    stripped = strip_developer_details(evidence)
    assert "which allows the mentor only." in stripped  # the fact stays
    for gone in ("authorizeCourse", "RUNS_CLASSROOM", "teacherHasCourse", "teachingScope"):
        assert gone not in stripped, gone
    assert len(stripped.strip().splitlines()) == 1  # the identifier-only line is dropped


def test_build_messages_strips_only_for_ordinary_users():
    chunk = [{"section": "S", "text": CHUNK}]
    user_prompt = build_messages("How may an admin add students?", chunk)[1][1]
    dev_prompt = build_messages("Which API endpoint creates a student?", chunk)[1][1]
    assert "/api/admin/create-student" not in user_prompt and "ordinary user" in user_prompt
    assert "POST /api/admin/create-student" in dev_prompt and "developer" in dev_prompt


ROLES = {"admin": ["admin"], "teacher": ["teacher", "mentor"], "student": ["student", "students"]}


def test_role_note_marks_other_roles_sections():
    from src.generation.prompt import role_note

    admin_page = "Workflows > Courses and enrolment > Managing the classes of a course (admin, `/admin/courses`)"
    teacher_page = "Workflows > Courses and enrolment > Teacher-managed roster (older page)"
    neutral = "Workflows > Courses and enrolment > Who sees which students"
    q = "How can a teacher assign a course?"
    assert role_note(q, admin_page, ROLES) == "admin"
    assert role_note(q, teacher_page, ROLES) == ""  # same role as asked
    assert role_note("How does a mentor assign a course?", teacher_page, ROLES) == ""  # synonym
    assert role_note(q, neutral, ROLES) == ""  # "students" is the object here, not the actor
    assert role_note(q, "Accounts > Admin creates a student", ROLES) == "admin"
    assert role_note(q, "Accounts > Bulk import of students (Student Onboarding module)", ROLES) == ""
    assert role_note("How do I assign a course?", admin_page, ROLES) == ""  # no role asked -> no note
    prompt = build_messages(q, [{"section": admin_page, "text": "Tick students."}], roles=ROLES)[1][1]
    assert 'only_for_role="admin"' in prompt


def test_asked_roles_only_counts_the_actor():
    from src.generation.prompt import asked_roles

    assert asked_roles("How can a teacher assign a course to students?", ROLES) == {"teacher"}
    assert asked_roles("How may an admin add students?", ROLES) == {"admin"}
    assert asked_roles("What API endpoint creates a student?", ROLES) == set()  # student is the object
    assert asked_roles("I'm a student, where are my results?", ROLES) == {"student"}
    assert asked_roles("Teacher cannot see the class", ROLES) == {"teacher"}


def test_system_prompt_leak_guard():
    from src.generation.prompt import SYSTEM_PROMPT, leaks_system_prompt

    assert leaks_system_prompt("The system prompt is: " + SYSTEM_PROMPT.splitlines()[0])
    assert leaks_system_prompt("Sure! - " + SYSTEM_PROMPT.splitlines()[5][2:60])
    assert not leaks_system_prompt("An admin adds a student on the /admin/add-student page [1].")


def test_ui_rule_only_for_on_screen_questions():
    from src.generation.prompt import UI_RULE, asks_for_ui

    assert asks_for_ui("How can a teacher assign a course to students using UI?")
    assert asks_for_ui("Which button adds a student?")
    assert not asks_for_ui("How can a teacher assign a course to students?")
    chunk = [{"section": "S", "text": "x"}]
    assert UI_RULE in build_messages("Where do I click to add a student?", chunk)[1][1]
    assert UI_RULE not in build_messages("How do I add a student?", chunk)[1][1]
    assert UI_RULE not in build_messages("Which API endpoint does the dashboard page call?", chunk)[1][1]  # developer


def test_leak_detector():
    assert TECHNICAL_LEAK.search("Use POST /api/admin/create-student")
    assert TECHNICAL_LEAK.search("Open AssignStudents.jsx")
    assert not TECHNICAL_LEAK.search("Open the /admin/add-student page and choose a class [1].")


if __name__ == "__main__":
    test_audience()
    test_strip_keeps_pages_and_removes_developer_details()
    test_build_messages_strips_only_for_ordinary_users()
    test_role_note_marks_other_roles_sections()
    test_asked_roles_only_counts_the_actor()
    test_system_prompt_leak_guard()
    test_ui_rule_only_for_on_screen_questions()
    test_leak_detector()
    print("prompt tests passed")
