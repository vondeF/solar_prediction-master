import asyncio
import aiohttp
import json
import pprint


async def someText():
    while True:
        print('Запись из другого модуля')
        await asyncio.sleep(5)


async def getTestForecast():
    n = 0
    async with aiohttp.ClientSession() as session:
        while True:
            n += 1
            async with session.get('http://127.0.0.1:5000/test_forecast') as response:
                #print('Результат запроса номер ' + str(n))
                print("Status:", response.status)
                print("Content-type:", response.headers['content-type'])
                html = await response.text()
                print("Body:\n", html)
                await asyncio.sleep(5)

async def main():
    loop.create_task(getTestForecast())
    loop.create_task(someText())

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.run_forever()