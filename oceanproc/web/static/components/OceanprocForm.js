import { html } from 'htm/preact';
import { render } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { FileBrowser } from './FileBrowser.js'

export function OceanprocForm(props) {
	const [displayRestOfForm, setDisplayRestOfForm] = useState(false)
	const [bidsPath, setBidsPath] = useState("")

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
		${displayRestOfForm ==- true  && (
			html`
					
			`
		)}
		</div>
	`
}
