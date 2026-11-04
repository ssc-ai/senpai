# astro-senpai has moved to senpai

SENPAI is now published on PyPI as [`senpai`](https://pypi.org/project/senpai/).
This package is empty and only installs the same version of `senpai`, so
existing `astro-senpai` installs and pins keep working. New installs should use:

```sh
pip install senpai
```

The import package is unchanged: `import senpai`.

## Upgrading from astro-senpai 2.9.0 or earlier with pip

Those releases contained the code themselves. When pip upgrades them to this
package it removes the old files after installing `senpai`, which deletes
`senpai`'s files too. If `senpai` is broken after an upgrade (for example,
`senpai.__version__` is missing), reinstall it:

```sh
pip install --force-reinstall --no-deps senpai
```

Or switch cleanly: `pip uninstall astro-senpai && pip install senpai`.
uv is not affected.

Source: <https://github.com/ssc-ai/senpai>
