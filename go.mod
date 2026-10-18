module github.com/gsoultan/mimir

go 1.27.0

require (
	github.com/ebitengine/purego v0.11.1
	github.com/gsoultan/mim v0.0.0-20261018040759-70ce158f529a
	github.com/klauspost/compress v1.20.3
)

// The mim module is served from the repository github.com/gsoultan/mmf
// until that repository is renamed to match its module path.
replace github.com/gsoultan/mim => github.com/gsoultan/mmf v0.0.0-20261018040759-70ce158f529a
