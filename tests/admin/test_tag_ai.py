"""Tag · AI shows what the AI saw (the model-input frames, labelled), "Full scene" marked not seen by the AI, a form
that follows the clip's prompt version (legacy box answer or the Eye's category form), and "In my words" -> Convert,
saved with the words, their language, the converting model and the prompt version."""
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from home_guard_project.admin.demo_backend import DemoBackend
from home_guard_project.admin.tag_view import FULL_SCENE, TagView

LEGACY_CLIP = 'of:production_demo/back_1791031000_alert'      # its meta: teacher.prompt_version "2026-10-03.demo"
EYE_CLIP = 'ds:yard_1791000007_trigger'                        # an old dataset clip no prompt answered


def tag_view(widgets, wait, key):
    b = DemoBackend()
    v = TagView(b, 'admin'); widgets.append(v); v.resize(1366, 768); v.show(); v.open(key)
    wait(lambda: v.key == key and v.detail is not None and not v.clip_runner.busy, 10)
    return v, b


def test_the_ai_view_steps_the_model_input_and_full_scene_is_marked(widgets, wait):
    v, _ = tag_view(widgets, wait, LEGACY_CLIP)
    wait(lambda: not v.media_runner.busy and v.ai_images, 10)
    assert v.view == 'crop' and v.segments['crop'].text() == 'AI view'
    assert v.view_label.text().startswith('What the AI sees: whole frame · 1 fps · 6 frames · 640×360')
    assert 'whole frames: no crop was saved' in v.view_label.text()
    assert v.canvas.image.size() == v.ai_images[0].size()
    v.setFocus(); QTest.keyClick(v, Qt.Key.Key_Period)
    assert v.ai_index == 1 and v.clock.text().startswith('frame 2 / 6')
    v.toggle_play(); assert v.ai_timer.isActive() and v.ai_timer.interval() == 1000
    v.toggle_play(); assert not v.ai_timer.isActive()
    v.switch_view('clip')
    assert v.view_label.text() == FULL_SCENE and not v.ai_images


def test_the_form_follows_the_prompt_version(widgets, wait):
    v, _ = tag_view(widgets, wait, LEGACY_CLIP)
    assert v.kind == 'legacy' and v.detail['answer_schema']['fields'][0] == 'summary'
    assert v.category_box.isHidden() and not v.parts['legacy'][0].isHidden() and v.parts['zone'][1].isHidden()
    assert 'Prompt version 2026-10-03.demo' in v.schema_note.text() and v.form['prompt_version'] == '2026-10-03.demo'
    v.open_key(EYE_CLIP); wait(lambda: v.key == EYE_CLIP and not v.clip_runner.busy, 10)
    assert v.kind == 'eye' and not v.category_box.isHidden() and v.parts['legacy'][0].isHidden()
    assert 'No prompt answered this clip' in v.schema_note.text()


def test_in_my_words_converts_and_saves_the_words_as_ground_truth(widgets, wait):
    v, b = tag_view(widgets, wait, LEGACY_CLIP)
    v.words.setPlainText('ראיתי גבר עם תיק ליד הגדר האחורית, הוא הציץ פנימה')
    v.setFocus(); QTest.keyClick(v, Qt.Key.Key_W)
    wait(lambda: not v.side_runner.busy and v.form.get('converted_by'), 10)
    f = v.form
    assert f['converted_by'].startswith('demo/gemini') and f['tagger_language'] == 'he'
    assert f['raw_label'] == 'suspicious' and f['why'] and f['summary_owner'].startswith('ראיתי')
    assert v.label_chips.value() == 'suspicious' and v.why.toPlainText() == f['why']
    assert 'Converted from your words' in v.suggest_note.text() and v.dirty()        # a suggestion: not saved yet
    v.save(); wait(lambda: not v.save_runner.busy, 10)
    assert v.save_state.text() == 'Saved', v.banner_text.text()
    saved = b.tagging_clip(LEGACY_CLIP)['tag']['fields']
    assert saved['tagger_words'].startswith('ראיתי') and saved['tagger_language'] == 'he'
    assert saved['converted_by'].startswith('demo/gemini') and saved['prompt_version'] == '2026-10-03.demo'
    assert saved['description'] and saved['label'] == 'suspicious'


def test_convert_needs_words(widgets, wait):
    v, _ = tag_view(widgets, wait, EYE_CLIP)
    v.convert()
    assert v.banner.isVisible() and 'Write what you saw' in v.banner_text.text() and not v.side_runner.busy
