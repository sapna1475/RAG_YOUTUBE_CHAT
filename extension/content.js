//This runs on the YouTube page and reads the video ID from the URL.
//and send the id to popup.js when asked

//listen for mesg from popup.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse)=>{
    
    if(request.type === "GET_VIDEO_ID"){

        
    // YouTube video URLs look like:
    // https://www.youtube.com/watch?v=5KmopXwjXik
    // URLSearchParams lets us easily extract the "v" parameter
        const urlParams = new URLSearchParams(window.location.search);
        const videoId = urlParams.get("v")

        //send the if back to popup.js else null
        sendResponse({ videoId : videoId})

    }

    //return true to keep msgs channel open
    // (required when using sendResponse asynchronously)
    return true;
});
