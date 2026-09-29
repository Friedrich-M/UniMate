"""Validation shared by Blender and the inference worker."""
def validate_clips(clips, signature):
    if not clips:
        raise ValueError("Add at least one prompt clip.")
    if len(clips) > 32:
        raise ValueError("Use at most 32 prompt clips.")
    previous = None
    for clip in clips:
        start, end = clip["start"], clip["end"]
        if not isinstance(start, int) or not isinstance(end, int) or end - start < 1:
            raise ValueError("Each clip must cover at least two integer frames.")
        if end - start + 1 > 600:
            raise ValueError("Use clips of at most 600 frames; split longer motions.")
        if previous is not None and start != previous + 1:
            raise ValueError("Prompt ranges must be contiguous and ordered, with no gaps or overlaps.")
        if not clip["prompt"].strip():
            raise ValueError("Each clip needs a motion description.")
        used = set()
        for ref in clip.get("references", []):
            frame = ref["frame"]
            if not start <= frame <= end:
                raise ValueError("Reference frame must lie inside its prompt clip.")
            if frame in used:
                raise ValueError("Only one pose reference is allowed at a given frame.")
            used.add(frame)
            if not ref.get("pose"):
                raise ValueError("Preview or match each reference, then Capture Current Pose before generating.")
            if ref["pose"].get("signature") != signature:
                raise ValueError("A captured pose belongs to a different rest skeleton. Capture it again.")
        previous = end
    if clips[-1]["end"] - clips[0]["start"] + 1 > 10000:
        raise ValueError("A generated sequence may contain at most 10,000 frames.")
    return clips
