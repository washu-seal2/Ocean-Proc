import { html } from 'htm/preact';
import { render } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { FileEntry } from './FileEntry.js';

export function FileBrowser(props) {
	const [filepaths, setFilepaths] = useState({})
	const [changedDir, setChangedDir] = useState('')
	const [curChosenDir, setCurChosenDir] = useState('')
	const [submittedDir, setSubmittedDir] = useState('')
	const buttonRef = useRef(null)

	useEffect(() => {
		const xhr = new XMLHttpRequest();
		xhr.open('GET', '/api/get_filesystem/')
		xhr.onload = () => {
			if (xhr.status === 200) {
				setFilepaths(JSON.parse(xhr.responseText))
			}
		}
		xhr.send()
	}, [])
	
	// Refresh on dir change
	useEffect(() => {
		if (changedDir !== '') {
			const xhr = new XMLHttpRequest();
			xhr.open('GET', `/api/get_filesystem/${changedDir}`)
			xhr.onload = () => {
				if (xhr.status === 200) {
					setFilepaths(JSON.parse(xhr.responseText))
				}
			}
			xhr.send()
			setChangedDir('')
			setCurChosenDir('')
		}
	}, [changedDir])

	// Enable submit button on chosen radio button
	useEffect(() => {
		if (buttonRef.current !== null){
			if (curChosenDir !== '') {
				buttonRef.current.disabled = false
			} else {
				buttonRef.current.disabled = true
			}
		}
	}, [curChosenDir])
	
	if (JSON.stringify(filepaths) !== '{}') {
		return html`
			<div class="modal fade" id="${props.modalId}" tabindex="-1" role="dialog" aria-labelledby="${props.modalId}Title" aria-hidden="true">
				<div class="modal-dialog" role="document">
					<div class="modal-content">
						<div class="modal-header text-light bg-primary">
							<h5 class="modal-title" id="${props.modalId}Title">${props.browserTitle}</h5>
							<button type="button" class="close" data-dismiss="modal" aria-label="Close">
          						<span aria-hidden="true">X</span>
        					</button>
						</div>
						<div class="modal-body text-light bg-dark">
							<ul class="fileEntryList" >
								${filepaths.file_list.map(file => html`
									<${FileEntry} file=${file} key=${file} setChangedDir=${setChangedDir} setCurChosenDir=${setCurChosenDir}><//>
								`)}
							</ul>
						</div>
						<div class="modal-footer">
							<button ref=${buttonRef} class="btn btn-primary" data-dismiss="modal" onClick=${() => props.handleSubmit(curChosenDir)} disabled>${props.submitBtnText}</button>
						</div>
					</div>
				</div>
			</div>
		`
	} else {
		return html`
			<div class="modal FileBrowserContainer" id="${props.modalId}" tabindex="-1" role="dialog" aria-labelledby="${props.modalId}Title" aria-hidden="true">
				<div class="spinner-border" role="status" style="width: 10rem; height: 10rem;">
					<span class="sr-only">Loading...</span>
				</div>
			</div>
		`
	}
}
