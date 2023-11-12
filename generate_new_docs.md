Keep automate_mkdocs.py, mkdocs.yml and mkgendocs.yml at src/
When new code is added at main, merge into automate_docs branch
Go to src/ and run the following commands in the terminal:

```{bash}
python automate_mkdocs.py
gendocs --config mkgendocs.yml
mkdocs build
```

Then move the "site" folder to the root of the repository and then run

```{bash}
mkdocs gh-deploy
```

Done. The new documentation is now available at
https://industry-time-series.github.io/industry-ts/