import dlib


def start_tracker(box, rgb, inputQueue, outputQueue):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)

    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        rgb = inputQueue.get()

        # if there was an entry in our queue, process it
        if rgb is not None:
            # update the tracker and grab the position of the tracked
            # object
            t.update(rgb)
            pos = t.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the label + bounding box coordinates to the output
            # queue
            outputQueue.put((startX, startY, endX, endY))
