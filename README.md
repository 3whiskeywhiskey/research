# research.whiskey.works

Source of the research notes site. Jekyll, built by GitHub Pages' own builder (no
Actions workflow needed), published at https://research.whiskey.works.

## Layout

- `_projects/<slug>.md`: one note per file. Front matter drives the index and the page
  head: `title`, `subtitle`, `date`, `updated`, `hardware`, `status`, `summary`, `hero`,
  `hero_caption`, `links` (label + url), `cite` (BibTeX). The body is Markdown; `##`
  headings feed the sticky contents list.
- `assets/projects/<slug>/`: figures for that note.
- `code/<slug>/`: scripts the note cites. Excluded from the site build, present in the
  repository.
- `_layouts`, `_includes`, `assets/css/site.css`: the site itself. `DESIGN.md` records
  the design choices (not published).
- `index.html`: the gallery. The newest note is featured with its hero figure; earlier
  notes go into the tile grid.

## Adding a note

1. Copy figures to `assets/projects/<slug>/` and scripts to `code/<slug>/`.
2. Write `_projects/<slug>.md` with the front matter above. `hero` should be a figure
   that reads at 1650x880 (it is cropped to that ratio on the index).
3. Build locally (below), check the index and the note, commit, push.

Corrections go in the note's "Changelog" section with a date, and `updated` in the front
matter moves; the original text stays.

## Local build

The exact Pages builder, no Ruby setup:

```
docker run --rm -v "$PWD":/src -w /src \
  -e GITHUB_WORKSPACE=/src -e INPUT_SOURCE=. -e INPUT_DESTINATION=./_site -e INPUT_FUTURE=false \
  -e INPUT_BUILD_REVISION= -e INPUT_VERBOSE=false -e INPUT_TOKEN= \
  -e GITHUB_REPOSITORY=3whiskeywhiskey/research -e PAGES_REPO_NWO=3whiskeywhiskey/research \
  -e JEKYLL_ENV=production ghcr.io/actions/jekyll-build-pages:v1.0.13
python3 -m http.server -d _site 4000
```

Or with Ruby: `bundle install && bundle exec jekyll serve` (the Gemfile pins the
`github-pages` gem set).

## One-time setup

1. Create the public repository `3whiskeywhiskey/research`, push `main`.
2. Repository settings, Pages: Source "Deploy from a branch", branch `main`, folder `/`.
   Custom domain `research.whiskey.works`; tick "Enforce HTTPS" once the certificate is
   issued (a few minutes after DNS resolves). The `CNAME` file in the repo keeps the
   domain set across pushes.
3. DNS: a `CNAME` record for `research.whiskey.works` pointing at
   `3whiskeywhiskey.github.io`. Do not resolve the name before the record exists.
4. Account settings, Pages: verify the `whiskey.works` domain so nobody else can bind
   subdomains of it to their Pages site.

## Licence

Text and figures CC BY 4.0. Code MIT unless a file's header says otherwise.
