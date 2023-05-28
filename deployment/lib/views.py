from aiohttp.web import Response
from aiohttp.web import View
from aiohttp_jinja2 import render_template


class IndexView(View):
    async def get(self) -> Response:
        return render_template("index.html", self.request, {})

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            text = form["text"]

            client = self.request.app["client"]
            model = self.request.app["model"]

            search_result = client.search(
                collection_name='test_collection',
                query_vector=list(map(float, model(text))), 
                limit=10
            )

            items = []
            for item in search_result:
                items.append(
                    {
                        "title": item.payload["title"],
                        "description": item.payload["description"],
                        "keywords": item.payload["keywords"],
                    }
                )
            ctx = {"text": text, "items": items}

        except Exception as exc:
            ctx = {"error": repr(exc)}

        return render_template("index.html", self.request, ctx)
