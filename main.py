"""AstrBot entrypoint. No message hooks or business work are registered."""

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
