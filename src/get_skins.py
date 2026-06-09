import asyncio
from io import BytesIO
import numpy as np

import aiohttp
from pathlib import Path
from PIL import Image


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
OUT_64_DIR = OUT_DIR / "64x64"
OUT_32_DIR = OUT_DIR / "64x32"
UUID_LIST = DATA_DIR / "uuids.txt"
LAST_BATCH_FILE = DATA_DIR / "last_batch.txt"

OUT_64_DIR.mkdir(exist_ok=True, parents=True)
OUT_32_DIR.mkdir(exist_ok=True, parents=True)


class Skin:

    def __init__(self):
        self.URL = "https://minotar.net"
        self.SKIN_URL = f"{self.URL}/skin"

    async def _get_skin(self, session: aiohttp.ClientSession, uuid: str) -> Image.Image:
        async with session.get(f"{self.SKIN_URL}/{uuid}") as response:
            response.raise_for_status()
            return Image.open(BytesIO(await response.read()))

    async def get_skin(self, uuid: str) -> Image.Image:
        async with aiohttp.ClientSession() as session:
            return await self._get_skin(session, uuid)

    async def get_skins(self, uuids: list[str]) -> list[Image.Image]:
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*(self._get_skin(session, uuid) for uuid in uuids), return_exceptions=True)

    def average_skins(self, skins: list[Image.Image]) -> Image.Image:
        arrays = [np.array(s) for s in skins]
        avg_array = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(avg_array)

    def save_skin(self, image: Image.Image, filename: str) -> Path:
        output_path = OUT_DIR / filename
        image.save(output_path)
        return output_path


async def main():
    api = Skin()
    batch_size = 100

    if LAST_BATCH_FILE.exists():
        batch_id = int(LAST_BATCH_FILE.read_text().strip()) + 1
        print(f"Resuming from batch #{batch_id}")
    else:
        batch_id = 0

    with open(UUID_LIST, "r") as f:
        for _ in range(batch_id * batch_size):
            f.readline()

        while True:
            print(f"Fetching batch #{batch_id}")
            uuids = [f.readline().strip() for _ in range(batch_size)]
            skins = await api.get_skins(uuids)

            print(f"Saving batch #{batch_id}")
            for i, skin in enumerate(skins):
                if skin.size == (64, 64):
                    skin.save(OUT_64_DIR / f"{uuids[i]}.webp")
                else:
                    skin.save(OUT_32_DIR / f"{uuids[i]}.webp")

            print(f"Completed batch #{batch_id}")

            LAST_BATCH_FILE.write_text(str(batch_id))
            batch_id += 1

if __name__ == "__main__":
    asyncio.run(main())
