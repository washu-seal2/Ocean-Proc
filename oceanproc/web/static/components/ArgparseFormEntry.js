import { html } from 'htm/preact';
import { useState, useEffect, useRef, useMemo } from 'preact/hooks';
import { FileBrowser } from './FileBrowser.js';

export function ArgparseFormEntry(props) {		
	const [optName, setOptName] = useState("")
	const [optObj, setOptObj] = useState({})
	const [optHtml, setOptHtml] = useState(html``)

	useEffect(() => {
		setOptName(Object.keys(props.obj)[0])
		setOptObj(props.obj[Object.keys(props.obj)[0]])
	}, [])

	useEffect(() => {
		switch (optObj.action) {
			case "store_true":
				setOptHtml(html`
					<div class="form-check">
						<input class="form-check-input" type=checkbox id="${optName}Input" name="${props.formName}"></input>
						<label class="form-check-label" for=${optName}>${optName}</label>
					</div>
				`)
				break;
			default:
				setOptHtml(html`
					<div class="form-group argparseFormTextInput">
						<label for=${optName}>${optName}</label>
						<input class="form-control" type=text id="${optName}Input" name="${props.formName}"></input>
					</div>
				`)
		}
		if (optObj.type !== undefined && optObj.type.includes("Path")) {
				setOptHtml(html`
					<div class="form-group argparseFormTextInput">
						<p>${optName}</p>
						<button type="button" 
						class="btn btn-primary" 
						data-toggle="modal" 
						data-target="#${optName}Modal">Select path for ${optName}</button>
						<${FileBrowser}
						modalId="${optName}Modal"
						submitBtnText="Choose directory"
						browserTitle="Select path for ${optName}"
						handleSubmit=${(path) => console.log(path)}><//>
					</div>
				`)
		}
	}, [optObj])
	return optHtml
}

