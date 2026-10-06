"""The API gateway: boxes ask our server for a model, the server asks the providers.

The provider keys stay on the server; a box holds only its own token. The server
maps a model alias (``eye``) to an ordered list of upstreams, falls back when one
fails, meters every call per box into SQLite, and refuses a box (HTTP 402) once it
is over its daily dollar cap. See README.md in this folder.
"""
