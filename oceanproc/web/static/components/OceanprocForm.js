import { html } from 'htm/preact';
import { render } from 'preact';
import { useState, useEffect, useRef, useMemo } from 'preact/hooks';
import { FileBrowser } from './FileBrowser.js'
import { ArgparseFormEntry } from './ArgparseFormEntry.js'

export function OceanprocForm(props) {
	const [displayRestOfForm, setDisplayRestOfForm] = useState(false)
	const [argparseArgObjects, setArgparseArgObjects] = useState(null)
	const [bidsPath, setBidsPath] = useState("")
	
	useEffect(() => {
		const xhr = new XMLHttpRequest();
		xhr.open('GET', '/api/get_parser_args/')
		xhr.onload = () => {
			if (xhr.status === 200) {
				setArgparseArgObjects(JSON.parse(xhr.responseText))
			}
		}
		xhr.send()
	}, [])
	useEffect(() => {
		if (bidsPath !== "") {
			setDisplayRestOfForm(true)
		} else {
			setDisplayRestOfForm(false)
		}
	}, [bidsPath])

	return html`
		<div class="formContainer">
			<button type="button" class="btn btn-primary" data-toggle="modal" data-target="#bidsDirForm">
				Select your BIDS directory
			</button>
			<p>${bidsPath}</p>
			<${FileBrowser} 
			modalId="bidsDirForm" 
			submitBtnText="Choose directory" 
			browserTitle="Choose a BIDS directory (click on a directory name to go into it, check the radio button to select it)"
			handleSubmit=${(curChosenDir) => setBidsPath(curChosenDir)}><//>
			<form>
				${displayRestOfForm && argparseArgObjects !== null && html`
					${argparseArgObjects.map(obj => html`
						<${ArgparseFormEntry} 
						obj=${obj} 
						formName="restOfForm"
						ignoredOptNames=${["bids_path"]}><//>
					`)}	
				`}
			</form>
		</div>
	`
}
