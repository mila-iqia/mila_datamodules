from .utils import replace_kwargs


def foo(a: int = 123, b: str = "abc"):
    print(a, b)
    return a, b


def test_replace_kwargs():
    fixed_foo = replace_kwargs(foo, a=456)
    assert fixed_foo() == (456, "abc")
