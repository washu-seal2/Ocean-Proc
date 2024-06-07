#!/bin/bash
reqs_txt=$(cat ../requirements.txt | sed -e 's/^/\t"/g' -e 's/$/",/g')
reqs_txt=$(printf "%s\n%s\n%s" 'dependencies = [' "$reqs_txt" ']')




