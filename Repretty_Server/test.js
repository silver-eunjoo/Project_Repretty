const response = await fetch('http://127.0.0.1/question', {
    method: 'POST',
    headers: {
        'Content-Type': 'text/event-stream'
    },
    body: JSON.stringify({
        "question": "갱년기라 힘들어"
    })
})
const reader = response.body.getReader();
while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    console.log('Received', new TextDecoder().decode(value));
}

/*
stream client test code.
js 기반으로는 잘 작동하는데 왜 다른건 안될까..
 */