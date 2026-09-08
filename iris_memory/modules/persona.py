"""Adoption of plugin-owned personas through the public AstrBot carrier API."""


class Module:
    def __init__(self, control):
        self.control = control

    async def start(self):
        await self.control.host.check_persona_support()

    async def close(self):
        await self.control.host.release_personas(self.control)

    async def prepare(self, identity):
        if not self.control.allowed(identity):
            return None
        persona = await self.control.personas.resolve(
            identity, self.control.settings["default_persona"]
        )
        if persona:
            await self.control.host.adopt_persona(self.control, identity, persona)
        else:
            await self.control.host.release_personas(self.control, identity.umo)
        return persona
