function getUrl(apiUrl: string, path: string = "", params: any = {}) {
    const url = new URL(`${apiUrl}/${path}`);
    url.search = new URLSearchParams(params).toString();
    return url
}

export function get(apiUrl: string, path: string = "", params: any = {}) {
    return fetch(getUrl(apiUrl, path, params)).then((res) => res.json())
}

export function post(apiUrl: string, path: string = "", params: any = {}, body: any = {}) {
    return fetch(getUrl(apiUrl, path, params), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    }).then((res) => res.json())
}
