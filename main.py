"""AstrBot public hooks; raw capture never calls a model or sends a message."""

from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools

from .iris_memory import PLUGIN_NAME
from .iris_memory.application import Application
from .iris_memory.pages.api import PagesAPI


class IrisMemory(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.application = None
        self.pages = None

    async def initialize(self):
        if self.application is not None:
            return
        application = Application(StarTools.get_data_dir(PLUGIN_NAME))
        self.application = application
        try:
            await application.initialize()
            self.pages = PagesAPI(application)
            self.pages.register(self.context)
        except BaseException:
            await application.terminate()
            raise

    async def terminate(self):
        if self.application is not None:
            await self.application.terminate()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def observe_message(self, event):
        if self.application and self.application.state == "ready":
            await self.application.delivery.capture(event)

    @filter.on_llm_response()
    async def observe_generated(self, event, response):
        if self.application and self.application.state == "ready":
            await self.application.delivery.observe_stage(event, "generated_callback")

    @filter.after_message_sent()
    async def observe_send_callback(self, event):
        if self.application and self.application.state == "ready":
            await self.application.delivery.observe_stage(event, "after_send_callback")
