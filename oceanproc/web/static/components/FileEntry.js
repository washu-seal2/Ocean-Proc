import { html } from 'htm/preact';
import { render } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { DirIcon } from './DirIcon.js'
import { FileIcon } from './FileIcon.js'

export function FileEntry(props) {
	function handleClick(path) {
		props.setChangedDir(path)		
	}

	if (props.file.isdir){
		return html`
			<div class="fileEntryContainer">
				<input type="radio" name="dirchoice" value=${props.file.realpath} onClick=${() => props.setCurChosenDir(props.file.realpath)}></input>
				<div class="fileLinkContainer" onClick=${() => handleClick(props.file.pathfromroot)}>
					<${DirIcon}><//>
					<li key=${props.key}>${props.file.basename}</li>
				</div>
			</div>
		`
	} else {
		return html`
			<div class="fileEntryContainer">
				<div class="fileNonlinkContainer">
					<${FileIcon}><//>
					<li key=${props.key}>${props.file.basename}</li>
				</div>
			</div>
		`
	}
}
