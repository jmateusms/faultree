from .data import gates


def _get_event_name(event):
    """
    Returns the name of the event
    """
    if 'id' in event:
        return event['id']
    else:
        return event[list(event.keys())[0]]


def _analyze_event(event):
    """
    Analyzes the event and returns the type of event and its value or children.
    """
    if 'children' in event and 'gate' in event:
        if event['children']:
            if event['gate'] in gates:
                return event['gate'], event['children']
            else:
                raise Exception(f'Invalid gate for event {event}.')
        elif event['gate'] in gates:
            raise Exception(f'Valid gate ({event["gate"]}), but no children found for event {_get_event_name(event)}.')

    if 'value' in event:
        if event['value'] is not None:
            return 'basic_event', event['value']
        else:
            raise Exception(f'No valid gate, children or value found for event {_get_event_name(event)}.')
    else:
        raise Exception(f'No valid gate, children or value found for event {_get_event_name(event)}.')


def eval_fta(fta):
    """
    Evaluates FTA
    """
    event_analysis = _analyze_event(fta)
    if event_analysis[0] == 'basic_event':
        result = event_analysis[1]
    else:
        result = 1
        if event_analysis[0] == 'AND':
            for child in fta['children']:
                result *= eval_fta(child)
        elif event_analysis[0] == 'OR':
            for child in fta['children']:
                result *= (1 - eval_fta(child))
            result = 1 - result

    return result


def eval_rta(rta):
    """
    Evaluates RTA (reliability tree)
    """
    event_analysis = _analyze_event(rta)
    if event_analysis[0] == 'basic_event':
        result = event_analysis[1]
    else:
        result = 1
        if event_analysis[0] == 'OR':
            for child in rta['children']:
                result *= eval_rta(child)
        elif event_analysis[0] == 'AND':
            for child in rta['children']:
                result *= (1 - eval_rta(child))
            result = 1 - result

    return result
