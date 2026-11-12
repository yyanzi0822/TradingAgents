from trading_engine.commands import HELP, CommandBot, parse_command
from trading_engine.killswitch import KillSwitch

OWNER = "9009868785"


def update(uid, text, chat=OWNER):
    return {"update_id": uid, "message": {"chat": {"id": int(chat)}, "text": text}}


class FakeTelegram:
    def __init__(self, *batches):
        self.batches = list(batches)
        self.params = []
        self.sent = []

    def get(self, url, params, timeout):
        self.params.append(dict(params))
        return {"ok": True, "result": self.batches.pop(0) if self.batches else []}

    def send(self, text):
        self.sent.append(text)
        return True


def bot(tmp_path, tg, status=lambda: "summary"):
    ks = KillSwitch(tmp_path / "KILL_SWITCH")
    return CommandBot("TOKEN", OWNER, ks, status, offset_path=tmp_path / "offset",
                      get=tg.get, send=tg.send), ks


def test_parse_command_accepts_slash_bot_suffix_and_chinese():
    assert parse_command("/stop earnings tomorrow") == ("stop", "earnings tomorrow")
    assert parse_command("/STOP@mibujiang_bot") == ("stop", "")
    assert parse_command("停止 太波动") == ("stop", "太波动")
    assert parse_command("恢复") == ("resume", "")
    assert parse_command("状态") == ("status", "")
    assert parse_command("hello") == (None, "")
    assert parse_command("") == (None, "")


def test_stop_and_resume_flip_the_kill_switch(tmp_path):
    tg = FakeTelegram([update(1, "/stop too volatile")], [update(2, "/resume")])
    b, ks = bot(tmp_path, tg)
    assert b.poll_once() == 1
    assert ks.active() and "too volatile" in ks.reason()
    assert "紧急开关已打开" in tg.sent[-1]
    b.poll_once()
    assert not ks.active()
    assert "已关闭" in tg.sent[-1]


def test_messages_from_anyone_else_are_ignored(tmp_path):
    tg = FakeTelegram([update(1, "/stop", chat="12345")])
    b, ks = bot(tmp_path, tg)
    assert b.poll_once() == 0
    assert not ks.active() and tg.sent == []


def test_offset_is_saved_so_a_command_is_handled_once(tmp_path):
    tg = FakeTelegram([update(41, "/stop")])
    b, _ = bot(tmp_path, tg)
    b.poll_once()
    b.poll_once()
    assert "offset" not in tg.params[0]
    assert tg.params[1]["offset"] == 42
    assert (tmp_path / "offset").read_text() == "42"
    assert len(tg.sent) == 1


def test_status_replies_and_survives_a_broker_failure(tmp_path):
    tg = FakeTelegram()
    b, ks = bot(tmp_path, tg, status=lambda: "Daily summary ok")
    assert b.handle("/status") == "Daily summary ok"

    def broken():
        raise ConnectionError("down")

    b2, ks2 = bot(tmp_path, tg, status=broken)
    ks2.engage("test")
    reply = b2.handle("状态")
    assert "ConnectionError" in reply and "打开" in reply


def test_unknown_text_gets_help(tmp_path):
    b, _ = bot(tmp_path, FakeTelegram())
    assert b.handle("hi") == HELP
